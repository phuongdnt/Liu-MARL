import time
# import wandb 
import os
import numpy as np
from itertools import chain
import torch
from pathlib import Path 
from tensorboardX import SummaryWriter

from utils.separated_buffer import SeparatedReplayBuffer
from utils.util import update_linear_schedule

def _t2n(x):
    return x.detach().cpu().numpy()

class CRunner(object):
    def __init__(self, config):

        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.n_render_rollout_threads = self.all_args.n_render_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval
        self.n_warmup_evaluations = self.all_args.n_warmup_evaluations
        self.n_no_improvement_thres = self.all_args.n_no_improvement_thres

        # dir
        self.model_dir = self.all_args.model_dir

        if self.use_wandb:
            import wandb
            self.save_dir = str(wandb.run.dir)
            self.run_dir = str(wandb.run.dir)
        else:
            self.run_dir = config["run_dir"]
            self.log_dir = str(self.run_dir / 'logs')
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writter = SummaryWriter(self.log_dir)
            self.save_dir = self.run_dir / 'models'

        from algorithms.happo_trainer import HAPPO as TrainAlgo
        from algorithms.happo_policy import HAPPO_Policy as Policy

        self.share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0]

        # policy network
        self.policy = []
        for agent_id in range(self.num_agents):
            po = Policy(self.all_args,
                        self.envs.observation_space[agent_id],
                        self.share_observation_space,
                        self.envs.action_space[agent_id],
                        device = self.device)
            self.policy.append(po)

        if self.model_dir is not None:
            self.restore()

        self.trainer = []
        self.buffer = []
        for agent_id in range(self.num_agents):
            tr = TrainAlgo(self.all_args, self.policy[agent_id], device = self.device)
            bu = SeparatedReplayBuffer(self.all_args,
                                       self.envs.observation_space[agent_id],
                                       self.share_observation_space,
                                       self.envs.action_space[agent_id])
            self.trainer.append(tr)
            self.buffer.append(bu)
            
    def run(self):
        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        best_reward = float('-inf')
        best_bw = []
        best_cost = []
        best_service = []
        
        record = 0

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)
                    
                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)

                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic 
                
                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()
            
            # post process
            self.total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                self.total_num_steps,
                                self.num_env_steps,
                                int(self.total_num_steps / (end - start))))

                if self.env_name == "MyEnv":
                    for agent_id in range(self.num_agents):
                        train_infos[agent_id].update({"average_step_rewards": np.mean(self.buffer[agent_id].rewards)})
                self.log_train(train_infos, self.total_num_steps)

            # evaluation
            if self.use_eval and episode % self.eval_interval == 0:
                re, cost_res, service_res, bw_res = self.eval(self.total_num_steps)
                
                if(re > best_reward and episode > self.n_warmup_evaluations):
                    print(f"New best reward: {re}. Saving model...")
                    self.save()
                    best_reward = re
                    best_bw = bw_res
                    best_cost = cost_res
                    best_service = service_res
                    record = 0
                else:
                    record += 1
                
                if(record == self.n_no_improvement_thres):
                    print("Training finished...")
                    return best_reward, best_bw, best_cost, best_service

        return best_reward, best_bw, best_cost, best_service

    def warm_up_eval(self):
        pass

    def warmup(self):
        # reset env
        obs = self.envs.reset()
        share_obs = []
        for o in obs:
            share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)

        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))
            self.buffer[agent_id].share_obs[0] = share_obs.copy()
            self.buffer[agent_id].obs[0] = np.array(list(obs[:, agent_id])).copy()

    def collect(self, step):
        values = []
        actions = []
        temp_actions_env = []
        action_log_probs = []
        rnn_states = []
        rnn_states_critic = []

        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            value, action, action_log_prob, rnn_state, rnn_state_critic \
                = self.trainer[agent_id].policy.get_actions(self.buffer[agent_id].share_obs[step],
                                                            self.buffer[agent_id].obs[step],
                                                            self.buffer[agent_id].rnn_states[step],
                                                            self.buffer[agent_id].rnn_states_critic[step],
                                                            self.buffer[agent_id].masks[step])
            values.append(_t2n(value))
            action = _t2n(action)
            if self.envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                for i in range(self.envs.action_space[agent_id].shape):
                    uc_action_env = np.eye(self.envs.action_space[agent_id].high[i]+1)[action[:, i]]
                    if i == 0:
                        action_env = uc_action_env
                    else:
                        action_env = np.concatenate((action_env, uc_action_env), axis=1)
            elif self.envs.action_space[agent_id].__class__.__name__ == 'Discrete':
                action_env = np.squeeze(np.eye(self.envs.action_space[agent_id].n)[action], 1)
            else:
                raise NotImplementedError

            actions.append(action)
            temp_actions_env.append(action_env)
            action_log_probs.append(_t2n(action_log_prob))
            rnn_states.append(_t2n(rnn_state))
            rnn_states_critic.append(_t2n(rnn_state_critic))

        actions_env = []
        for i in range(self.n_rollout_threads):
            one_hot_action_env = []
            for temp_action_env in temp_actions_env:
                one_hot_action_env.append(temp_action_env[i])
            actions_env.append(one_hot_action_env)

        values = np.array(values).transpose(1, 0, 2)
        actions = np.array(actions).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_probs).transpose(1, 0, 2)
        rnn_states = np.array(rnn_states).transpose(1, 0, 2, 3)
        rnn_states_critic = np.array(rnn_states_critic).transpose(1, 0, 2, 3)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        share_obs = []
        for o in obs:
            share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)

        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))

            self.buffer[agent_id].insert(share_obs,
                                        list(obs[:, agent_id]),
                                        rnn_states[:, agent_id],
                                        rnn_states_critic[:, agent_id],
                                        actions[:, agent_id],
                                        action_log_probs[:, agent_id],
                                        values[:, agent_id],
                                        rewards[:, agent_id],
                                        masks[:, agent_id])

    def compute(self):
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            next_value = self.trainer[agent_id].policy.get_values(self.buffer[agent_id].share_obs[-1], 
                                                                self.buffer[agent_id].rnn_states_critic[-1],
                                                                self.buffer[agent_id].masks[-1])
            next_value = _t2n(next_value)
            self.buffer[agent_id].compute_returns(next_value, self.trainer[agent_id].value_normalizer)

    def train(self):
        train_infos = []
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_training()
            train_info = self.trainer[agent_id].train(self.buffer[agent_id])
            train_infos.append(train_info)       
            self.buffer[agent_id].after_update()

        return train_infos

    def save(self):
        import os
        from pathlib import Path
        
        save_path = Path(self.save_dir).resolve()
        if not save_path.exists():
            try:
                os.makedirs(str(save_path), exist_ok=True)
            except Exception as e:
                print(f"[WARNING] Không thể tạo thư mục {save_path}: {e}")

        for agent_id in range(self.num_agents):
            policy_actor = self.trainer[agent_id].policy.actor
            actor_file = save_path / f"actor_agent{agent_id}.pt"
            
            try:
                with open(actor_file, 'wb') as f:
                    torch.save(policy_actor.state_dict(), f)
            except Exception as e:
                print(f"[ERROR] Lỗi khi lưu Actor {agent_id}: {e}")
            
            policy_critic = self.trainer[agent_id].policy.critic
            critic_file = save_path / f"critic_agent{agent_id}.pt"
            
            try:
                with open(critic_file, 'wb') as f:
                    torch.save(policy_critic.state_dict(), f)
            except Exception as e:
                print(f"[ERROR] Lỗi khi lưu Critic {agent_id}: {e}")

    def restore(self):
        for agent_id in range(self.num_agents):
            policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor_agent' + str(agent_id) + '.pt')
            self.policy[agent_id].actor.load_state_dict(policy_actor_state_dict)
            policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic_agent' + str(agent_id) + '.pt')
            self.policy[agent_id].critic.load_state_dict(policy_critic_state_dict)

    def log_train(self, train_infos, total_num_steps):
        for agent_id in range(self.num_agents):
            for k, v in train_infos[agent_id].items():
                agent_k = "agent%i/" % agent_id + k
                if self.use_wandb:
                    import wandb
                    wandb.log({agent_k: v}, step=total_num_steps)
                else:
                    self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        for k, v in env_infos.items():
            if len(v) > 0:
                if self.use_wandb:
                    import wandb
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)

    def eval(self, total_num_steps):
        eval_episode_rewards = []
        # [FIX] Quan trọng: reset với train=False để tính toán chỉ số Cost/Service
        self.eval_envs.reset(train=False)

        eval_obs = self.eval_envs.get_reset_obs(True)

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            eval_temp_actions_env = []
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_rollout()
                eval_action, eval_rnn_state = self.trainer[agent_id].policy.act(np.array(list(eval_obs[:, agent_id])),
                                                                                eval_rnn_states[:, agent_id],
                                                                                eval_masks[:, agent_id],
                                                                                deterministic=True)

                eval_action = eval_action.detach().cpu().numpy()
                if self.eval_envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                    for i in range(self.eval_envs.action_space[agent_id].shape):
                        eval_uc_action_env = np.eye(self.eval_envs.action_space[agent_id].high[i]+1)[eval_action[:, i]]
                        if i == 0:
                            eval_action_env = eval_uc_action_env
                        else:
                            eval_action_env = np.concatenate((eval_action_env, eval_uc_action_env), axis=1)
                elif self.eval_envs.action_space[agent_id].__class__.__name__ == 'Discrete':
                    eval_action_env = np.squeeze(np.eye(self.eval_envs.action_space[agent_id].n)[eval_action], 1)
                else:
                    raise NotImplementedError

                eval_temp_actions_env.append(eval_action_env)
                eval_rnn_states[:, agent_id] = _t2n(eval_rnn_state)
            
            eval_actions_env = []
            for i in range(self.n_eval_rollout_threads):
                eval_one_hot_action_env = []
                for eval_temp_action_env in eval_temp_actions_env:
                    eval_one_hot_action_env.append(eval_temp_action_env[i])
                eval_actions_env.append(eval_one_hot_action_env)

            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
        
        cost_res = self.eval_envs.get_eval_cost_res()
        service_res = self.eval_envs.get_eval_service_res()
        bw_res = self.eval_envs.get_eval_bw_res()

        re = np.mean(np.sum(eval_episode_rewards, axis=0))
        print("Eval average reward: ", re, 
              " Eval cost: ", cost_res, 
              " Eval service: ", service_res, 
              " Eval bw: ", bw_res)

        return re, cost_res, service_res, bw_res
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from envs.net_2x3 import Env

class MultiDiscrete(gym.Space):
    def __init__(self, array_of_param_array):
        super().__init__()
        self.low = np.array([x[0] for x in array_of_param_array])
        self.high = np.array([x[1] for x in array_of_param_array])
        self.num_discrete_space = self.low.shape[0]
        self.n = np.sum(self.high) + 2

    def sample(self):
        random_array = np.random.rand(self.num_discrete_space)
        return [int(x) for x in np.floor(np.multiply((self.high - self.low + 1.), random_array) + self.low)]

    def contains(self, x):
        return len(x) == self.num_discrete_space and (np.array(x) >= self.low).all() and (
                    np.array(x) <= self.high).all()

    @property
    def shape(self):
        return self.num_discrete_space

    def __repr__(self):
        return "MultiDiscrete" + str(self.num_discrete_space)

    def __eq__(self, other):
        return np.array_equal(self.low, other.low) and np.array_equal(self.high, other.high)


class SubprocVecEnv(object):
    def __init__(self, all_args):
        self.env_list = [Env() for i in range(all_args.n_rollout_threads)]
        self.num_envs = all_args.n_rollout_threads
        self.num_agent = self.env_list[0].agent_num
        self.signal_obs_dim = self.env_list[0].obs_dim
        self.signal_action_dim = self.env_list[0].action_dim
        self.u_range = 1.0
        self.movable = True
        self.discrete_action_space = True
        self.discrete_action_input = False
        self.force_discrete_action = False

        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        share_obs_dim = 0
        for agent in range(self.num_agent):
            total_action_space = []
            if self.discrete_action_space:
                u_action_space = spaces.Discrete(self.signal_action_dim)
            else:
                u_action_space = spaces.Box(low=-self.u_range, high=+self.u_range, shape=(2,), dtype=np.float32)
            if self.movable:
                total_action_space.append(u_action_space)

            if len(total_action_space) > 1:
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])

            share_obs_dim += self.signal_obs_dim
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(self.signal_obs_dim,),
                                                     dtype=np.float32))

        self.share_observation_space = [spaces.Box(low=-np.inf, high=+np.inf, shape=(share_obs_dim,),
                                                   dtype=np.float32) for _ in range(self.num_agent)]

    def step(self, actions):
        results = [env.step(action) for env, action in zip(self.env_list, actions)]
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def get_property(self):
         inv = [env.get_inventory() for env in self.env_list]
         demand = [env.get_demand() for env in self.env_list]
         orders = [env.get_orders() for env in self.env_list]
         return inv, demand, orders
         
    def reset(self, train=True):
        obs = [env.reset(train=train) for env in self.env_list]
        return np.stack(obs)

    def close(self):
        pass

    def render(self, mode="rgb_array"):
        pass


# single env
class DummyVecEnv(object):
    def __init__(self, all_args):
        self.env_list = [Env() for i in range(all_args.n_rollout_threads)]
        self.num_envs = all_args.n_rollout_threads

        self.num_agent = self.env_list[0].agent_num
        self.signal_obs_dim = self.env_list[0].obs_dim
        self.signal_action_dim = self.env_list[0].action_dim

        self.u_range = 1.0
        self.movable = True
        self.discrete_action_space = True
        self.discrete_action_input = False
        self.force_discrete_action = False

        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        share_obs_dim = 0
        for agent_num in range(self.num_agent):
            total_action_space = []
            if self.discrete_action_space:
                u_action_space = spaces.Discrete(self.signal_action_dim)
            else:
                u_action_space = spaces.Box(low=-self.u_range, high=+self.u_range, shape=(2,), dtype=np.float32)
            if self.movable:
                total_action_space.append(u_action_space)

            if len(total_action_space) > 1:
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])

            obs_dim = self.signal_obs_dim
            share_obs_dim += obs_dim
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))

        self.share_observation_space = [spaces.Box(low=-np.inf, high=+np.inf, shape=(share_obs_dim,),
                                                   dtype=np.float32) for _ in range(self.num_agent)]

    def step(self, actions):
        results = [env.step(action) for env, action in zip(self.env_list, actions)]
        obs, rews, dones, infos = zip(*results)
        
        obs = np.stack(obs)
        rews = np.stack(rews)
        dones = np.stack(dones)
        
        # [FIX QUAN TRỌNG] Auto-Reset để tránh lỗi IndexError khi training
        for i, done in enumerate(dones):
            if np.any(done):
                # Reset về chế độ mặc định (train=True) để tiếp tục vòng lặp
                obs[i] = self.env_list[i].reset() 
                
        return obs, rews, dones, infos

    def reset(self, train=True):
        # [FIX] Nhận tham số train để chuyển đổi chế độ Train/Eval
        obs = [env.reset(train=train) for env in self.env_list]
        return np.stack(obs)
    
    def get_eval_bw_res(self):
        res = self.env_list[0].get_eval_bw_res()
        return res
    
    def get_eval_cost_res(self):
        res = self.env_list[0].get_eval_cost_res()
        return res

    def get_eval_service_res(self):
        res = self.env_list[0].get_eval_service_res()
        return res

    def get_eval_num(self):
        eval_num = self.env_list[0].get_eval_num()
        return eval_num
        
    def close(self):
        pass

    def render(self, mode="rgb_array"):
        pass
    
    def get_reset_obs(self, normalize=True):
        obs = [env.get_reset_obs(normalize) for env in self.env_list]
        return np.stack(obs)
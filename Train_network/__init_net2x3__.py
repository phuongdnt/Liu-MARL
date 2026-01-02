
import socket
from absl import flags
FLAGS = flags.FLAGS
FLAGS(['train_sc.py'])

from .serial import Env as SerialEnv
from .net_2x3 import Env as NetworkEnv

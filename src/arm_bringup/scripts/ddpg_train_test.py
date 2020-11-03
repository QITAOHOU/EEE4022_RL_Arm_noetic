from tf2rl.algos.ddpg import DDPG
from tf2rl.experiments.trainer import Trainer
from arm_env import ArmEnvironment
import rospy
import numpy as np

parser = Trainer.get_argument()
parser = DDPG.get_argument(parser)
#parser.set_defaults(max_steps=5)
parser.set_defaults(model_dir='model_DDPG_static')
#parser.set_defaults(dir_suffix='/home/devon/RL_Arm_noetic/')
parser.set_defaults(normalise_obs=False)
parser.set_defaults(save_model_interval=1000)
#parser.set_defaults(show_progress=True)
args = parser.parse_args()
rospy.init_node('test')


env = ArmEnvironment(static_goal=True)
test_env = ArmEnvironment(static_goal=True)
print("Obs shape:",env.observation_space.shape)
print("action shape:",env.action_space.high.size)
policy = DDPG(
    state_shape=env.observation_space.shape,
    action_dim=env.action_space.high.size,
    gpu=0,  # Run on CPU. If you want to run on GPU, specify GPU number
    memory_capacity=10000,
    max_action=np.array([0.5,0.5,0.5,0.5]),
    batch_size=32,
    n_warmup=0)
trainer = Trainer(policy, env, args, test_env=test_env)
print("args:",args)
#trainer._set_check_point('results')
trainer()
#trainer.evaluate_policy_continuously()
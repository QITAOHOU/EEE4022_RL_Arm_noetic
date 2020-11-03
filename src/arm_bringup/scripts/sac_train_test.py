from tf2rl.algos.sac import SAC
from tf2rl.experiments.trainer import Trainer
from arm_env import ArmEnvironment
import rospy
import numpy as np


parser = Trainer.get_argument()
parser = SAC.get_argument(parser)
#parser.set_defaults(max_steps=5)
parser.set_defaults(model_dir='model_SAC_static')
#parser.set_defaults(dir_suffix='/home/devon/RL_Arm_noetic/')
parser.set_defaults(normalise_obs=False)
parser.set_defaults(save_model_interval=1000)
parser.set_defaults(gpu=0)
parser.set_defaults(max_steps=160000)

#parser.set_defaults(show_progress=True)
args = parser.parse_args()
rospy.init_node('RL_agent')


env = ArmEnvironment(static_goal=True)
test_env = ArmEnvironment(static_goal=True)

policy = SAC(
        state_shape=env.observation_space.shape,
        action_dim=env.action_space.high.size,
        gpu=args.gpu,
        memory_capacity=args.memory_capacity,
        max_action=env.action_space.high[0],
        batch_size=args.batch_size,
        n_warmup=0,
        alpha=args.alpha,
        auto_alpha=args.auto_alpha)
trainer = Trainer(policy, env, args, test_env=test_env)
trainer()
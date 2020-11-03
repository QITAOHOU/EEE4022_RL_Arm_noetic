from tf2rl.algos.ppo import PPO
from tf2rl.experiments.on_policy_trainer import OnPolicyTrainer
from tf2rl.envs.utils import is_discrete, get_act_dim
from arm_env import ArmEnvironment
import rospy
import numpy as np

rospy.init_node("RL_agent")
parser = OnPolicyTrainer.get_argument()
parser = PPO.get_argument(parser)

parser.set_defaults(model_dir='model_PPO_static')
parser.set_defaults(normalise_obs=False)
parser.set_defaults(save_model_interval=1000)
parser.set_defaults(horizon=35)
parser.set_defaults(batch_size=5)
parser.set_defaults(gpu=0)
args = parser.parse_args()

env = ArmEnvironment(static_goal=True)
test_env = ArmEnvironment(static_goal=True)

policy = PPO(
        state_shape=env.observation_space.shape,
        action_dim=get_act_dim(env.action_space),
        is_discrete=is_discrete(env.action_space),
        max_action=None if is_discrete(
            env.action_space) else env.action_space.high[0],
        batch_size=args.batch_size,
        actor_units=(64, 64),
        critic_units=(64, 64),
        n_epoch=10,
        lr_actor=3e-4,
        lr_critic=3e-4,
        hidden_activation_actor="tanh",
        hidden_activation_critic="tanh",
        discount=0.99,
        lam=0.95,
        entropy_coef=0.,
        horizon=args.horizon,
        normalize_adv=args.normalize_adv,
        enable_gae=args.enable_gae,
        gpu=args.gpu)
trainer = OnPolicyTrainer(policy, env, args, test_env=test_env)
trainer()

rospy.spin()
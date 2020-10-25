
from arm_env import ArmEnvironment
import numpy as np
import rospy

env = ArmEnvironment()
state_shape = env.state_shape
action_shape = env.action_shape

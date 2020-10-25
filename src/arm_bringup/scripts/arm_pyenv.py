from arm_env import AllJoints, ArmEnvironment
from tf_agents.environments import py_environment
import numpy as np
import rospy

class ArmEnv(py_environment.PyEnvironment):
    def __init__(self):
        self.ros_env = ArmEnvironment()
        self.ros_env.set_new_goal()

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.float64, minimum=-0.5, maximum=0.5, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(7,), dtype=np.float64, minimum=[-1.5,-1.5,-1.5,-2.5,-0.4,-0.4,0], maximum=[1.5,1.5,1.5,2.5,0.4,0.4,0.4], name='observation')
        self._state = np.concatenate([np.zeros(4),self.ros_env.get_goal_pos()]
        self._episode_ended = False
    
    def action_spec(self):
        return self._action_spec
    
    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._episode_ended = False


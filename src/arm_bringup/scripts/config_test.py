import rospy
from arm_env import ArmEnvironment, AllJoints
import numpy as np

rospy.init_node("test")

env = ArmEnvironment(static_goal=False)
env.reset()
# env.__init__(static_goal=False)

# env.set_new_goal()
# rospy.sleep(2)
# env.set_new_goal()
# rospy.sleep(2)
# env.set_new_goal()
# rospy.sleep(2)


# env.unpause_physics()


alljoints = AllJoints(env.joint_names)

pos = np.array([1,1,1,1])
alljoints.move(pos)
rospy.sleep(3)

# alljoints.move(pos)
# #rospy.sleep(3)
# env.pause_physics()
# env.unpause_physics()

# env.set_model_config()

# env.pause_physics()

# alljoints.move(np.zeros(4))
# rospy.sleep(1)
# env.unpause_physics()


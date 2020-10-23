import rospy
from arm_env import ArmEnvironment, AllJoints
import position_command
import actionlib
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import JointTrajectoryControllerState, FollowJointTrajectoryAction, FollowJointTrajectoryActionGoal, FollowJointTrajectoryGoal
from std_srvs.srv import Empty
from sensor_msgs.msg import JointState
from gazebo_msgs.srv import SetModelState, SetModelStateRequest, SetModelConfiguration, SetModelConfigurationRequest
from gazebo_msgs.srv import GetModelState, GetModelStateRequest
from gazebo_msgs.msg import ModelState
import numpy as np
envboi = ArmEnvironment()
envboi.reset()
zero = np.array([0,0,0,0])

testpos = np.array([0,0.9,0.5,1.5])



#rospy.init_node('testing_node')

# jointboi = AllJoints(['plat_joint','shoulder_joint','forearm_joint','wrist_joint'])

# jointboi.move(testpos)

# rospy.sleep(1)

# TODO fix reset function

# rospy.sleep(1)
# print(envboi.get_latest_state())
# envboi.reset()
# rospy.sleep(2)
# #rospy.spin()


#position_command.pos_commander(testpos)
rewards = np.array([])
while True:
    current_distance = envboi.get_goal_distance()
    print(("Current distance is: "+str(current_distance)))
    action = (input("Enter an action from -1 to 1: \n"))
    action = action.split(',')
    action = [float(i) for i in action]
    action = np.array(action)
    print(action)
    (state, reward, done) = envboi.step(action)
    "Reward is {}, Done: {}".format(reward, done)
    rewards = np.append(rewards, [reward])
    if(done):
        total_rewards = str(np.sum(rewards))
        print(("Well done! you reached the goal \n Total reward: "+total_rewards))
        break
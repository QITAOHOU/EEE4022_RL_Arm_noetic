import rospy
import actionlib
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import JointTrajectoryControllerState, FollowJointTrajectoryAction, FollowJointTrajectoryActionGoal, FollowJointTrajectoryGoal
from std_srvs.srv import Empty
from sensor_msgs.msg import JointState
from gazebo_msgs.srv import SetModelState, SetModelStateRequest, SetModelConfiguration, SetModelConfigurationRequest
from gazebo_msgs.srv import GetModelState, GetModelStateRequest
from gazebo_msgs.msg import ModelState
import numpy as np

def pos_commander(poslist):
    pub = rospy.Publisher('/arm_controller/command', JointTrajectory, queue_size=10)

    rate = rospy.Rate(10)  # 10hz
    pos_cmd = JointTrajectory()
    pos_cmd.joint_names = ['plat_joint','shoulder_joint','forearm_joint','wrist_joint']

    pt = JointTrajectoryPoint()

    pt.positions = poslist

    pt.time_from_start = rospy.Duration(2, 0)

    pos_cmd.points.append(pt)
    rospy.sleep(1)  # Needed so that the subscriber has enough time to subscribe to the publisher before publishing
    pub.publish(pos_cmd)

""" if __name__ == "__main__":
    poslist = list(input("Enter Position \n"))
    print(type(poslist))
    poslist = [float(i) for i in poslist] 
    print(poslist)
    pos_commander(poslist) """
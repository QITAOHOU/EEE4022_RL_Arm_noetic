import rospy
from control_msgs.msg import JointTrajectoryControllerState

def callback(data):
    rospy.loginfo("The current state is: %s", data.actual.positions)

def listener():
    state_sub = rospy.Subscriber('/arm_controller/state', JointTrajectoryControllerState, callback)
    rospy.sleep(2)
    rospy.spin()

def getLatestState():
    latest_state = rospy.wait_for_message('/arm_controller/state', JointTrajectoryControllerState)
    return latest_state.actual.positions

""" if __name__ == "__main__":
    listener() """
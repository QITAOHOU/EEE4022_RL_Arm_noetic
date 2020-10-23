import position_command
import current_position
import numpy as np
import rospy

poslist = np.linspace(2.0,0.0,15)
rospy.init_node('joint_control')

for i in poslist:

    current_pos = list(current_position.getLatestState())
    new_pos = [i-0.3 for i in current_pos]
    position_command.pos_commander(new_pos)
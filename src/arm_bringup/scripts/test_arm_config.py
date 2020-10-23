import rospy
from controller_manager_msgs.srv import SwitchController, SwitchControllerRequest
from gazebo_msgs.srv import SetModelState, SetModelStateRequest, SetModelConfiguration, SetModelConfigurationRequest
import numpy as np
rospy.init_node("test")

joint_names = ['plat_joint','shoulder_joint','forearm_joint','wrist_joint']
starting_pos = np.array([0, 0, 1, 2])

model_config_proxy = rospy.ServiceProxy('/gazebo/set_model_configuration',SetModelConfiguration)
model_config_req = SetModelConfigurationRequest()
model_config_req.model_name = 'arm'
model_config_req.urdf_param_name = 'robot_description'
model_config_req.joint_names = joint_names
model_config_req.joint_positions = starting_pos

controller_switch_proxy = rospy.ServiceProxy('/controller_manager/switch_controller',SwitchController)
controller_switch_off = SwitchControllerRequest()
controller_switch_off.stop_controllers = ['joint_state_controller','arm_controller']
controller_switch_off.strictness = 2

controller_switch_on = SwitchControllerRequest()
controller_switch_on.start_controllers = ['joint_state_controller','arm_controller']
controller_switch_on.strictness = 2

rospy.wait_for_service('/controller_manager/switch_controller')
try:
   controller_switch_proxy(controller_switch_off)
except rospy.ServiceException as e:
    print('/controller_manager/switch_controller call failed')


rospy.wait_for_service('/gazebo/set_model_configuration')
try:
    model_config_proxy(model_config_req)
    print("REEEEEEEEEEEEE")
except rospy.ServiceException as e:
    print('/gazebo/set_model_configuration call failed')

rospy.wait_for_service('/controller_manager/switch_controller')
try:
    controller_switch_proxy(controller_switch_on)
except rospy.ServiceException as e:
    print('/controller_manager/switch_controller call failed')



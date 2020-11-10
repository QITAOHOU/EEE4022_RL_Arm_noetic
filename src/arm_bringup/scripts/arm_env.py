import os
import gym
from gym import spaces
import rospy
import actionlib
from geometry_msgs.msg import Pose
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import JointTrajectoryControllerState, FollowJointTrajectoryAction, FollowJointTrajectoryActionGoal, FollowJointTrajectoryGoal
from std_srvs.srv import Empty
from sensor_msgs.msg import JointState
from gazebo_msgs.srv import SetModelState, SetModelStateRequest, SetModelConfiguration, SetModelConfigurationRequest
from gazebo_msgs.srv import GetModelState, GetModelStateRequest
from controller_manager_msgs.srv import SwitchController, SwitchControllerRequest, LoadController, LoadControllerRequest
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SpawnModel, DeleteModel, SpawnModelRequest
from rosgraph_msgs.msg import Clock
import tf
import tf2_ros
from scipy.spatial import distance

import numpy as np
import time

arm_model_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], '..', '..', 'arm_description',
                              'urdf', 'arm_gazebo.urdf')

sphere_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], '..', '..', 'arm_description',
                              'urdf', 'ball.urdf')


class AllJoints:
    def __init__(self,joint_names):
        self.action_server_client = actionlib.SimpleActionClient('arm/arm_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
       # rospy.loginfo('Waiting for joint trajectory action')
        self.action_server_client.wait_for_server()
       # rospy.loginfo('Found joint trajectory action')
        self.jtp = rospy.Publisher('arm/arm_controller/command', JointTrajectory, queue_size=1)
        self.joint_names = joint_names
        self.jtp_zeros = np.zeros(len(joint_names))


    def move(self, pos):
        msg = FollowJointTrajectoryActionGoal()
        msg.goal.trajectory.joint_names = self.joint_names
        point = JointTrajectoryPoint()
        point.positions = pos
        point.time_from_start = rospy.Duration(1.0/60.0)
        msg.goal.trajectory.points.append(point)
        self.action_server_client.send_goal(msg.goal)
        return True

    def move_jtp(self, pos):
        jtp_msg = JointTrajectory()
        jtp_msg.joint_names = self.joint_names
        point = JointTrajectoryPoint()
        point.positions = pos
        point.velocities = self.jtp_zeros
        point.accelerations = self.jtp_zeros
        point.effort = self.jtp_zeros
        point.time_from_start = rospy.Duration(1.0/60.0)
        jtp_msg.points.append(point)
        self.jtp.publish(jtp_msg)

    def reset_move(self, pos):
        jtp_msg = JointTrajectory()
        self.jtp.publish(jtp_msg)
        msg = FollowJointTrajectoryActionGoal()
        msg.goal.trajectory.joint_names = self.joint_names
        point = JointTrajectoryPoint()
        point.positions = pos
        point.time_from_start = rospy.Duration(0.0001)
        msg.goal.trajectory.points.append(point)
        self.action_server_client.send_goal(msg.goal)

    def reset_move_jtp(self):
        jtp_msg = JointTrajectory()
        jtp_msg.joint_names = self.joint_names
        point = JointTrajectoryPoint()
        point.positions = self.jtp_zeros
        point.time_from_start = rospy.Duration(1.0/60.0)
        jtp_msg.points.append(point)
        self.jtp.publish(jtp_msg)

class ArmEnvironment(gym.Env):
    def __init__(self, static_goal, slow_step=False, larger_radius=False):
        self.max_sim_time = 15
        if(larger_radius):
            self.goal_radius = 0.06 #sphere radius in ball.urdf is slightly bigger to improve visuals
        else:
            self.goal_radius = 0.05
        self.distance_reward_coeff = 5
        self.static_goal_pos = np.array([0.2,0.1,0.15])
        self.static_goal = static_goal
        self.slow_step = slow_step
        self.eef_pos = np.zeros(3)

        self.zero = np.array([0,0,0,0])
        #rospy.init_node('joint_position_node')
        self.num_joints = 4
        self.observation_space = spaces.Box(np.array([-1.5,-1.5,-1.5,-2.5,-0.4,-0.4,0]), np.array([1.5,1.5,1.5,2.5,0.4,0.4,0.4]))#(self.num_joints + 3,)
        self.action_space = spaces.Box(np.array([-0.2,-0.2,-0.2,-0.2]), np.array([0.2,0.2,0.2,0.2]))
        self.joint_names = ['plat_joint','shoulder_joint','forearm_joint','wrist_joint']
        self.all_joints = AllJoints(self.joint_names)
        self.starting_pos = np.array([0, 0, 0, 0])
        self.last_goal_distance = 0
        rospy.loginfo("Defining a goal position...")
        if(self.static_goal):
            self.goal_pos = self.static_goal_pos
        else:
            while(True):
                x_y = np.random.uniform(low = -0.4, high = 0.4, size = 2)
                z = np.random.uniform(low = 0, high = 0.4, size = 1)
                self.goal_pos = np.concatenate([x_y,z],axis=0)
                if(np.linalg.norm(self.goal_pos)<0.4):
                    break
        rospy.loginfo("Goal position defined")
        self.pause_physics_proxy = rospy.ServiceProxy('/gazebo/pause_physics',Empty, persistent=False)
        self.unpause_physics_proxy = rospy.ServiceProxy('/gazebo/unpause_physics',Empty, persistent=False)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty, persistent=False)
        #self.config_proxy = rospy.ServiceProxy('/gazebo/set_model_configuration', SetModelConfiguration)
        # self.config_request = SetModelConfigurationRequest()
        # self.config_request.model_name = 'arm'
        # self.config_request.urdf_param_name = 'arm/robot_description'
        # self.config_request.joint_names = self.joint_names
        # self.config_request.joint_positions = self.starting_pos
   
        self.load_controller_proxy = rospy.ServiceProxy('/arm/controller_manager/load_controller', LoadController, persistent=False)
        self.joint_state_controller_load = LoadControllerRequest()
        self.joint_state_controller_load.name = 'joint_state_controller'
        self.arm_controller_load = LoadControllerRequest()
        self.arm_controller_load.name = 'arm_controller'

        self.switch_controller_proxy = rospy.ServiceProxy('/arm/controller_manager/switch_controller', SwitchController, persistent=False)
        self.switch_controller = SwitchControllerRequest()
        self.switch_controller.start_controllers.append('joint_state_controller')
        self.switch_controller.start_controllers.append('arm_controller')
        self.switch_controller.strictness = 2

        self.del_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel, persistent=False)
        self.spawn_model_proxy = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel, persistent=False)
        self.arm_urdf = open(arm_model_dir, "r").read()
        self.arm_model = SpawnModelRequest()
        self.arm_model.model_name = 'arm'  # the same with sdf name
        self.arm_model.model_xml = self.arm_urdf
        self.arm_model.robot_namespace = 'arm'
        self.initial_pose = Pose()
        self.initial_pose.position.z = 0.0305
        self.arm_model.initial_pose = self.initial_pose 
        self.arm_model.reference_frame = 'world'

        self.state_proxy = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState, persistent=False)
        self.config_request = SetModelStateRequest()
        self.config_request.model_state.model_name = 'simple_ball'
        sphere_pose = Pose()
        sphere_pose.position.x = 1
        sphere_pose.position.y = 2
        sphere_pose.position.z = 3
        self.config_request.model_state.pose = sphere_pose
        self.config_request.model_state.reference_frame = 'world'
        # self.config_request.urdf_param_name = 'arm/robot_description'
        # self.config_request.joint_names = self.joint_names
        # self.config_request.joint_positions = self.starting_pos
        self.sphere_urdf = open(sphere_dir, "r").read()
        self.sphere = SpawnModelRequest()
        self.sphere.model_name = 'simple_ball'  # the same with sdf name
        self.sphere.model_xml = self.sphere_urdf
        self.sphere.robot_namespace = 'arm'
        self.sphere_initial_pose = Pose()
        self.sphere_initial_pose.position.x = self.goal_pos[0]
        self.sphere_initial_pose.position.y = self.goal_pos[1]
        self.sphere_initial_pose.position.z = self.goal_pos[2]
        self.sphere.initial_pose = self.sphere_initial_pose 
        self.sphere.reference_frame = 'world'
        # rospy.wait_for_service('/gazebo/spawn_urdf_model')
        # try:
        #     self.spawn_model_proxy(self.sphere.model_name, self.sphere.model_xml, self.sphere.robot_namespace, self.sphere.initial_pose, 'world')
        # except (rospy.ServiceException) as e:
        #     print("/gazebo/failed to build the target")
        self.unpause_physics()
        #rospy.wait_for_service('/gazebo/delete_model')
        # self.del_model('arm')
        #self.del_model('simple_ball')

        self.spawn_model(self.sphere)
        #self.tf_listener = tf.TransformListener()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        
        
        self.joint_pos_high = np.array([1.5, 1.5, 1.5, 2.5])
        self.joint_pos_low = np.array([-1.5, -1.5, -1.5, -2.5])
        self.joint_pos_range = self.joint_pos_high-self.joint_pos_low
        self.joint_pos_mid = self.joint_pos_range/2.0
        self.joint_pos = np.zeros(4)
        self.joint_state = np.zeros(self.num_joints)
        self.joint_state_subscriber = rospy.Subscriber('arm/arm_controller/state', JointTrajectoryControllerState, self.joint_state_subscriber_callback, queue_size=1)
        self.normed_sp = self.normalize_joint_state(self.starting_pos)
        self.clock_subscriber = rospy.Subscriber('/clock', Clock, self.clock_subscriber_callback, queue_size=1)
        self.hit_floor = False
        # rospy.wait_for_service('gazebo/reset_simulation')
        # try:
        #     self.reset_proxy()
        # except (rospy.ServiceException) as e:
        #     print("gazebo/reset_simulation service call failed")


    def set_model_state(self, set_state_request):
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            self.state_proxy(set_state_request)
            return True
        except rospy.ServiceException as e:
            print('/gazebo/set_model_state service call failed')
            return False

    def clock_subscriber_callback(self, clock):
        self.current_time = clock.clock.secs

    def normalize_joint_state(self, joint_pos):
        # TODO implement normalization
        return joint_pos

    def get_state(self):
        self.hit_floor = False
        joint_angles = self.joint_state
        current_sim_time = self.current_time
        trans = self.tf_buffer.lookup_transform('world', 'wrist_link', rospy.Time())
        rospy.sleep(0.1)
        end = self.tf_buffer.lookup_transform('world', 'dummy_eef', rospy.Time())
        if(trans.transform.translation.z<=0.02 or end.transform.translation.z<=0.01): 
            self.hit_floor=True
        if(self.get_goal_distance()<=self.goal_radius):
            arrived = True
        else: 
            arrived=False
        if(self.hit_floor or current_sim_time>=self.max_sim_time):
            time_runout = True
            print("ran out of time or hit floor")
        else:
            time_runout=False
        return joint_angles, time_runout, arrived

    def joint_state_subscriber_callback(self, joint_state):
        self.joint_state = np.array(joint_state.actual.positions)
        

    def pause_physics(self):
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause_physics_proxy()
            return True
        except rospy.ServiceException as e:
            print('/gazebo/pause_physics service call failed')
            return False

    def unpause_physics(self):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause_physics_proxy()
        except rospy.ServiceException as e:
            print('/gazebo/unpause_physics service call failed')

    def get_goal_distance(self):
        try:
            trans = self.tf_buffer.lookup_transform('world', 'dummy_eef', rospy.Time())
            x = trans.transform.translation.x
            y = trans.transform.translation.y
            z = trans.transform.translation.z
            trans = np.array([x,y,z])
            self.eef_pos = trans
            #print("EEF Position: {}".format(trans))
            
            goal_distance = distance.euclidean(trans,self.goal_pos)
            #print("Goal is at: {} \n".format(np.array2string(self.goal_pos)))
            return goal_distance
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            print("tf lookupTransform error")

    def set_new_goal(self):
        rospy.loginfo("Defining a goal position...")
        if(self.static_goal):
            self.goal_pos = self.static_goal_pos
        else:
            # x_y = np.random.uniform(low = -0.4, high = 0.4, size = 2)
            # z = np.random.uniform(low = 0, high = 0.4, size = 1)
            # self.goal_pos = np.concatenate([x_y,z],axis=0)
            while(True):
                x_y = np.random.uniform(low = -0.4, high = 0.4, size = 2)
                z = np.random.uniform(low = 0, high = 0.4, size = 1)
                self.goal_pos = np.concatenate([x_y,z],axis=0) #np.array([-0.13242582 , 0.29086919 , 0.20275278])
                if(np.linalg.norm(self.goal_pos)<0.4 and np.linalg.norm(self.goal_pos)>0.1):
                    break
        #rospy.loginfo("Goal position defined")
        #rospy.loginfo("Goal position: "+str(self.goal_pos))
        # self.sphere_urdf = open(sphere_dir, "r").read()
        # self.sphere = SpawnModelRequest()
        # self.sphere.model_name = 'simple_ball'  # the same with sdf name
        # self.sphere.model_xml = self.sphere_urdf
        # self.sphere.robot_namespace = 'arm'
        # self.sphere_initial_pose = Pose()
        # self.sphere_initial_pose.position.x = self.goal_pos[0]
        # self.sphere_initial_pose.position.y = self.goal_pos[1]
        # self.sphere_initial_pose.position.z = self.goal_pos[2]
        # self.sphere.initial_pose = self.sphere_initial_pose 
        # self.sphere.reference_frame = 'world'
        # self.spawn_model(self.sphere)
        sphere_pose = Pose()
        sphere_pose.position.x = self.goal_pos[0]
        sphere_pose.position.y = self.goal_pos[1]
        sphere_pose.position.z = self.goal_pos[2]
        self.config_request.model_state.pose = sphere_pose
        self.set_model_state(self.config_request)

    def spawn_model(self, model):
        rospy.wait_for_service('/gazebo/spawn_urdf_model')
        try:
            self.spawn_model_proxy(model.model_name, model.model_xml, model.robot_namespace, model.initial_pose, 'world')
        except (rospy.ServiceException) as e:
            print("/gazebo/failed to build the target")
        self.unpause_physics()

    def get_reward(self, time_runout, arrive):
        reward = -1*self.distance_reward_coeff*self.get_goal_distance()
        if(self.hit_floor):
            reward = reward - 50.0
        if(time_runout and not arrive):
            reward = reward - 25.0
        if(arrive):
            print("Arrived at goal")
            reward += 25.0
        return reward
        
    def reset(self):

        rospy.wait_for_service('/gazebo/delete_model')
        self.del_model('arm')
        #self.del_model('simple_ball')
        rospy.sleep(0.5)
        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")
        self.spawn_model(self.arm_model)

        #rospy.sleep(0.5)
        #self.spawn_model(self.arm_model)
        #rospy.sleep(1)
        rospy.wait_for_service('arm/controller_manager/load_controller')
        try:
            self.load_controller_proxy(self.joint_state_controller_load)
        except (rospy.ServiceException) as e:
            print('arm/controller_manager/load_controller service call failed')
        
        rospy.wait_for_service('arm/controller_manager/load_controller')
        try:
            self.load_controller_proxy(self.arm_controller_load)
        except (rospy.ServiceException) as e:
            print('arm/controller_manager/load_controller service call failed')

        rospy.wait_for_service('arm/controller_manager/switch_controller')
        try:
            self.switch_controller_proxy(self.switch_controller)
        except (rospy.ServiceException) as e:
            print('arm/controller_manager/switch_controller service call failed')


        self.set_new_goal()
        rospy.sleep(1)
       # rospy.sleep(3)
        # self.last_goal_distance = self.get_goal_distance()
        done = False
        joint_angles, time_runout, arrived = self.get_state()
        #self.last_joint = self.joint_state
        #self.last_pos = pos
        # diff_joint = np.zeros(self.num_joints)
        state = np.concatenate([joint_angles,self.goal_pos])#.reshape(1, -1)
        state = np.asarray(state, dtype = np.float32)
        self.joint_pos = np.zeros(4)

        return state
    def seed(self,seed):
        pass
        
    def step(self, action):
        action = np.array(action)

        self.joint_pos = np.clip(self.joint_pos + action,a_min=self.joint_pos_low,a_max=self.joint_pos_high)
        self.all_joints.move(self.joint_pos)
        if(self.slow_step):
            rospy.sleep(0.35)
        else:
            rospy.sleep(0.01)
        (joint_angles, time_runout, arrived) = self.get_state()
        reward = self.get_reward(time_runout, arrived)
        if(time_runout or arrived):
            done = True
        else:
            done = False
        state = np.concatenate([joint_angles,self.goal_pos])#.reshape(1, -1)
        state = np.asarray(state, dtype = np.float32)

        return state, reward, done, {}


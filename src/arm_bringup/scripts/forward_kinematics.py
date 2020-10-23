from ikpy.chain import Chain
import numpy as np
import matplotlib.pyplot
from mpl_toolkits.mplot3d import Axes3D
active_links = [False, True, False, True, True, True, False, False]

arm_chain = Chain.from_urdf_file("arm_gazebo.urdf",base_elements=["base_link"],last_link_vector=np.array([0, 0, 0.045]),active_links_mask=active_links)

print(arm_chain)

transformation_mat = arm_chain.forward_kinematics(np.ones(8))

end_pos = transformation_mat[:3, 3]

print(end_pos)

arm_pos = np.array([0,0,45,45])

ax = matplotlib.pyplot.figure().add_subplot(111, projection='3d')

arm_chain.plot(np.array([0,arm_pos[0],0,arm_pos[1],arm_pos[2],arm_pos[3],0,0]), ax)
matplotlib.pyplot.show()

arm_chain.plot()


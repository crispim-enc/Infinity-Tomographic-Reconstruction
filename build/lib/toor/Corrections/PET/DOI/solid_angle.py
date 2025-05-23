import numpy as np
import matplotlib.pyplot as plt
width = 1-np.abs(np.arange(-1,1,0.1))
height = 1-np.abs(np.arange(-1,1,0.1))

# distance_to_anh = np.arange(1,60)
distance_to_anh = np.abs(np.arange(-30,30,0.1))+30

width, height, distance_to_anh = np.meshgrid(width,height,distance_to_anh)
# distance_to_anh_other_side = np.abs(np.arange(1,60)-60)
distance_to_anh_other_side = np.abs(distance_to_anh-60)
solid_angle = (4*np.arctan(width*height/(2*distance_to_anh*np.sqrt(4*distance_to_anh**2+width**2+height**2))))



solid_angle_other_side = 4*np.arctan(width*height/(2*distance_to_anh_other_side*np.sqrt(4*distance_to_anh_other_side**2+width**2+height**2)))
# solid_angle_other_side = solid_angle_other_side/np.max(solid_angle_other_side)
p = solid_angle*solid_angle
p = p / np.max(p)
distance_to_center = 30-np.abs(distance_to_anh-30)
beam_proba = np.sqrt(width**2+height**2+distance_to_center**2) / np.sqrt(30**2+2**2+2**2)

fig = plt.figure()

plt.imshow(p[:,10,100:-100])
# fig = plt.figure()
# plt.imshow(beam_proba[:,10,100:-100])
# ax = plt.axes(projection='3d')
# ax.scatter3D(width,height,p)
plt.show()



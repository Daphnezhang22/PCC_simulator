import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from vpython import *
from alterable_grow import *


dp_lines_exp_14 =     pd.read_excel('D:/Spiral/Spiral_7_8_ok_del.xlsx', header=None ,engine='openpyxl')

dp_lines_simu = pd.read_excel('D:/Spiral/Spiral_7_8_sim_change12_45_smallerDia.xls', header=None )


#process raw data
m,n = dp_lines_exp_14.shape
dp_exp_processed_1 = np.zeros([int(m/3),3, n])
for i in range(int(m/3)):
    for j in range(n):
        dp_exp_processed_1[i, 0, j] = dp_lines_exp_14[j][i*3]  # X
        dp_exp_processed_1[i, 1, j] = dp_lines_exp_14[j][i*3+1]  # Y
        dp_exp_processed_1[i, 2, j] = dp_lines_exp_14[j][i*3+2] # Z

dp_exp_processed_1 = dp_exp_processed_1/10



############ Module 2   ###########

mod2_org_1_1 = vec(dp_exp_processed_1[0, :, 6][0], dp_exp_processed_1[0, :, 6][1], dp_exp_processed_1[0, :, 6][2])
mod2_org_1_2 = vec(dp_exp_processed_1[0, :, 7][0], dp_exp_processed_1[0, :, 7][1], dp_exp_processed_1[0, :, 7][2])
mod2_org_1_3 = vec(dp_exp_processed_1[0, :, 8][0], dp_exp_processed_1[0, :, 8][1], dp_exp_processed_1[0, :, 8][2])
mod2_org_1_o = (mod2_org_1_1 + mod2_org_1_2 + mod2_org_1_3)/3

### to be done marker wrong marker
mod2_end_1 = vec(dp_exp_processed_1[0, :, 3][0], dp_exp_processed_1[0, :, 3][1], dp_exp_processed_1[0, :, 3][2])
mod2_end_2 = vec(dp_exp_processed_1[0, :, 4][0], dp_exp_processed_1[0, :, 4][1], dp_exp_processed_1[0, :, 4][2])
mod2_end_3 = vec(dp_exp_processed_1[0, :, 5][0], dp_exp_processed_1[0, :, 5][1], dp_exp_processed_1[0, :, 5][2])
mod2_end_o = (mod2_end_1 + mod2_end_2 + mod2_end_3)/3


vo1 = mod2_org_1_1 - mod2_org_1_2
vo2 = mod2_org_1_3 - mod2_org_1_2
head_v_org = vector.cross( vo1.hat, vo2.hat)

ve1 = mod2_end_1 - mod2_end_2
ve2 = mod2_end_3 - mod2_end_2
end_head_v_begin = vector.cross( ve1.hat, ve2.hat)

calibation_rotation_matrix = get_rotation_matrix_from_vector(head_v_org , end_head_v_begin)


Mod2_end_1_o = mod2_end_1 - mod2_end_o
Mod2_end_2_o = mod2_end_2 - mod2_end_o
Mod2_end_3_o = mod2_end_3 - mod2_end_o



v1_Ang_plane_to_v1 = Cal_two_vector_angle_2pi(Mod2_end_1_o.hat, Mod2_end_1_o)
v2_Ang_plane_to_v1 = Cal_two_vector_angle_2pi(Mod2_end_2_o.hat, Mod2_end_1_o)
v3_Ang_plane_to_v1 = Cal_two_vector_angle_2pi(Mod2_end_3_o.hat, Mod2_end_1_o)


len_array_module2_act1_inverse = np.zeros([int(m/3),3])
for i in range(int(m/3)):


    mod2_org_1 = vec(dp_exp_processed_1[i, :, 6][0], dp_exp_processed_1[i, :, 6][1], dp_exp_processed_1[i, :, 6][2])
    mod2_org_2 = vec(dp_exp_processed_1[i, :, 7][0], dp_exp_processed_1[i, :, 7][1], dp_exp_processed_1[i, :, 7][2])
    mod2_org_3 = vec(dp_exp_processed_1[i, :, 8][0], dp_exp_processed_1[i, :, 8][1], dp_exp_processed_1[i, :, 8][2])
    mod2_org_o = (mod2_org_1 + mod2_org_2 + mod2_org_3)/3

    mod2_end_1 = vec(dp_exp_processed_1[i,:,3][0], dp_exp_processed_1[i,:,3][1], dp_exp_processed_1[i,:,3][2])
    mod2_end_2 = vec(dp_exp_processed_1[i,:,4][0], dp_exp_processed_1[i,:,4][1], dp_exp_processed_1[i,:,4][2])
    mod2_end_3 = vec(dp_exp_processed_1[i,:,5][0], dp_exp_processed_1[i,:,5][1], dp_exp_processed_1[i,:,5][2])
    mod2_end_o = (mod2_end_1 + mod2_end_2 + mod2_end_3) / 3


    v1 = mod2_end_1 - mod2_end_o
    v2 = mod2_end_2 - mod2_end_o
    v3 = mod2_end_3 - mod2_end_o

    ve12 = mod2_end_1 - mod2_end_2
    ve32 = mod2_end_3 - mod2_end_2
    head_end_v = vector.cross(ve12.hat, ve32.hat)

    vo12 = mod2_org_1 - mod2_org_2
    vo32 = mod2_org_3 - mod2_org_2
    head_org_v = vector.cross(vo12.hat, vo32.hat)

    theta = np.arccos(vector.dot(head_org_v.hat, head_end_v.hat))
    head_end_v_comp_org = head_org_v.hat*head_end_v.mag*np.cos(theta)
    head_end_v_comp_plan= head_end_v - head_end_v_comp_org
    head_Ang_org_plane_to_v1 = Cal_two_vector_angle_2pi(head_end_v_comp_plan, (mod2_org_1-mod2_org_o))


    r_softArm = 2 #(6 + 1 + 0.2 * 2) / 2
    dis = (mod2_end_o - mod2_org_o).mag
    R_center = (dis / 2) / np.sin(theta / 2)
    print("Mod2: theta={}, head_Ang_xy_plane_to_X={}, R_center={} ".format(np.rad2deg(theta),
                                                        np.rad2deg(head_Ang_org_plane_to_v1), R_center))


    len_array_module2_act1_inverse[i, 0] = (R_center - r_softArm * np.cos(
        v1_Ang_plane_to_v1 + head_Ang_org_plane_to_v1)) * theta
    len_array_module2_act1_inverse[i, 1] = (R_center - r_softArm * np.cos(
        v2_Ang_plane_to_v1 + head_Ang_org_plane_to_v1)) * theta
    len_array_module2_act1_inverse[i, 2] = (R_center - r_softArm * np.cos(
        v3_Ang_plane_to_v1 + head_Ang_org_plane_to_v1)) * theta








################# Module 1  base #############

## set 0 pressure position
bottom_1 = vec(dp_exp_processed_1[0,:,0][0], dp_exp_processed_1[0,:,0][1], 0)
bottom_2 = vec(dp_exp_processed_1[0,:,1][0], dp_exp_processed_1[0,:,1][1], 0)
bottom_3 = vec(dp_exp_processed_1[0,:,2][0], dp_exp_processed_1[0,:,2][1], 0)

org_1 = vec(dp_exp_processed_1[0,:,0][0], dp_exp_processed_1[0,:,0][1], dp_exp_processed_1[0,:,0][2])
org_2 = vec(dp_exp_processed_1[0,:,1][0], dp_exp_processed_1[0,:,1][1], dp_exp_processed_1[0,:,1][2])
org_3 = vec(dp_exp_processed_1[0,:,2][0], dp_exp_processed_1[0,:,2][1], dp_exp_processed_1[0,:,2][2])
'''
bottom_1 = vec(-2.72950486,  -2.88259507, 0)
bottom_2 = vec(3.3170852 ,  -1.14768076, 0)
bottom_3 = vec(-1.13336025,   3.29686977, 0)

org_1 = vec( -2.72950486,  -2.88259507, 22.64235431)
org_2 = vec( 3.3170852 ,  -1.14768076, 22.98367915)
org_3 = vec( -1.13336025,   3.29686977, 22.25610357)
'''

vo1 = org_1 - org_2
vo2 = org_3 - org_2
offset_Z_1_2 =  org_2.z - org_1.z
offset_Z_1_3 =  org_3.z - org_1.z
head_v_org = vector.cross( vo1.hat, vo2.hat)
head_v_vertical = vec(0,0,-1)

calibation_rotation_matrix = get_rotation_matrix_from_vector(head_v_org , head_v_vertical)


X_axis = vec(1,0,0)

v1_Ang_xy_plane_to_X = Cal_two_vector_angle_2pi(bottom_1.hat, bottom_1.hat)
v2_Ang_xy_plane_to_X = Cal_two_vector_angle_2pi(bottom_2.hat, bottom_1.hat)
v3_Ang_xy_plane_to_X = Cal_two_vector_angle_2pi(bottom_3.hat, bottom_1.hat)

############  Module 1   ###########
len_array_module1_act1_inverse = np.zeros([int(m/3),3])
for i in range(int(m/3)):
    current_1 = vec(dp_exp_processed_1[i,:,0][0], dp_exp_processed_1[i,:,0][1], dp_exp_processed_1[i,:,0][2])
    current_2 = vec(dp_exp_processed_1[i,:,1][0], dp_exp_processed_1[i,:,1][1], dp_exp_processed_1[i,:,1][2])
    current_3 = vec(dp_exp_processed_1[i,:,2][0], dp_exp_processed_1[i,:,2][1], dp_exp_processed_1[i,:,2][2])
    current_o = (current_1 + current_2 + current_3) / 3

    v1 = current_1 - current_o
    v2 = current_2 - current_o
    v3 = current_3 - current_o

    v12 = current_1 - current_2
    v32 = current_3 - current_2

    head_v = vector.cross(v12.hat, v32.hat)
    head_v_new_array = np.matmul(calibation_rotation_matrix, head_v.hat.value)
    head_v_new = vec(head_v_new_array[0], head_v_new_array[1], head_v_new_array[2])
    # a.dot( b)/(np.linalg.norm(a)*np.linalg.norm(b))
    head_v_xy_plane = vec(head_v_new.x, head_v_new.y, 0).hat
    #head_Ang_xy_plane_to_X = np.arccos(vector.dot(head_v_xy_plane, X_axis))
    #sgn = vector.cross(X_axis, head_v_xy_plane).z  # anticlock wise is +

    head_Ang_xy_plane_to_X = Cal_two_vector_angle_2pi(head_v_xy_plane, bottom_1.hat)  # np.mod(head_Ang_xy_plane_to_X * (-1)**(sgn < 0), 2 * pi)

    theta = np.arccos(vector.dot(head_v_new, head_v_vertical))
    theta = Cal_two_vector_angle_2pi(head_v_new, head_v_vertical)
    r_softArm = 3 #(6 + 1 + 0.2 * 2) / 2
    dis = (current_o - vec(0, 0, 0)).mag

    R_center = (dis / 2) / np.sin(theta / 2)
    print("Mod1: theta={}, head_Ang_xy_plane_to_X={}, R_center={} ".format(np.rad2deg(theta),
                                                        np.rad2deg(head_Ang_xy_plane_to_X), R_center))

    len_array_module1_act1_inverse[i, 0] = (R_center - r_softArm * np.cos(
        v1_Ang_xy_plane_to_X + head_Ang_xy_plane_to_X)) * theta
    len_array_module1_act1_inverse[i, 1] = (R_center - r_softArm * np.cos(
        v2_Ang_xy_plane_to_X + head_Ang_xy_plane_to_X)) * theta
    len_array_module1_act1_inverse[i, 2] = (R_center - r_softArm * np.cos(
        v3_Ang_xy_plane_to_X + head_Ang_xy_plane_to_X)) * theta

len_array_module1_act1_inverse[0,:] = len_array_module1_act1_inverse[1,:]


df1 = pd.DataFrame(len_array_module1_act1_inverse)
df2 = pd.DataFrame(len_array_module2_act1_inverse)
df = pd.concat([df1, df2], axis=1)
#df = pd.DataFrame(np.concatenate((len_array_module1_act1_inverse,len_array_module2_act1_inverse),axis = 1) )
df.to_excel('D:/Santanna/3D Grow/Trial_5/len_array_module1_act1_inverse.xls', index=False, header=False)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('X axis (cm)')
ax.set_ylabel('Y axis (cm)')
ax.set_zlabel('Z axis (cm)')
#for j in range(6):
#    ax.scatter(dp_exp_processed_1[:, 0, j], dp_exp_processed_1[:, 1, j], dp_exp_processed_1[:, 2, j]*-1, label ='experiment_%d'%j) #, c='turquoise'

center1_x = (dp_exp_processed_1[:, 0, 0] + dp_exp_processed_1[:, 0, 1] + dp_exp_processed_1[:, 0, 2])/3
center1_y = (dp_exp_processed_1[:, 1, 0] + dp_exp_processed_1[:, 1, 1] + dp_exp_processed_1[:, 1, 2])/3
center1_z = (dp_exp_processed_1[:, 2, 0] + dp_exp_processed_1[:, 2, 1] + dp_exp_processed_1[:, 2, 2])/3

center2_x = (dp_exp_processed_1[:, 0, 3] + dp_exp_processed_1[:, 0, 4] + dp_exp_processed_1[:, 0, 5])/3
center2_y = (dp_exp_processed_1[:, 1, 3] + dp_exp_processed_1[:, 1, 4] + dp_exp_processed_1[:, 1, 5])/3
center2_z = (dp_exp_processed_1[:, 2, 3] + dp_exp_processed_1[:, 2, 4] + dp_exp_processed_1[:, 2, 5])/3

#ax.scatter(center1_x, center1_y, center1_z*-1, label ='experiment_m1', s = 2)
ax.scatter(center2_x, center2_y, center2_z*-1, label ='experiment_m2', s = 2)
#ax.plot(center2_x, center2_y, center2_z*-1, label ='experiment2_c')



dp_lines_simu = dp_lines_simu*10

theta = np.deg2rad(100+180)
for i in range(2):
    new_x = dp_lines_simu[0 + i * 3].mul(np.cos(theta)) - dp_lines_simu[1 + i * 3].mul( np.sin(theta) )
    new_y = dp_lines_simu[0 + i * 3].mul(np.sin(theta)) + dp_lines_simu[1 + i * 3].mul( np.cos(theta) )
    dp_lines_simu[0 + i * 3] = new_x
    dp_lines_simu[1 + i * 3] = new_y


for i in range(2):
    dp_lines_simu[0 + i * 3] = dp_lines_simu[0 + i * 3] + 0 # move to center
    dp_lines_simu[1 + i * 3] = dp_lines_simu[1 + i * 3] - 0.75
    dp_lines_simu[2 + i * 3] = dp_lines_simu[2 + i * 3] - 2

'''
x_sim = pd.concat([dp_lines_simu[0], dp_lines_simu[3]])#,  dp_lines_simu[6]])
y_sim = pd.concat([dp_lines_simu[1], dp_lines_simu[4]])#,  dp_lines_simu[7]])
z_sim = pd.concat([dp_lines_simu[2], dp_lines_simu[5]])#,  dp_lines_simu[8]])
'''

'''
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('X axis (cm)')
ax.set_ylabel('Y axis (cm)')
ax.set_zlabel('Z axis (cm)')
'''
#for j in range(6):
#    ax.scatter(dp_exp_processed_1[:, 0, j], dp_exp_processed_1[:, 1, j], dp_exp_processed_1[:, 2, j]*-1, c='turquoise', label ='experiment_'+j)

#ax.scatter(x_sim, y_sim, z_sim, c='skyblue',  label = 'simulation')

#ax.scatter(x_sim, y_sim, z_sim, c='skyblue',  label = 'simulation')
#ax.plot(dp_lines_simu[0], dp_lines_simu[1], dp_lines_simu[2], c='skyblue',  label = 'simulation')
#ax.scatter(dp_lines_simu[3], dp_lines_simu[4], dp_lines_simu[5],  label = 'simulation_m1',s = 3)
ax.scatter(dp_lines_simu[0], dp_lines_simu[1], dp_lines_simu[2],  label = 'simulation_m2',s = 3)
ax.legend()
plt.show()




#exit()


############ connection 1  ###############
org_1 = np.array([dp_exp_processed_1[0,:,0][0], dp_exp_processed_1[0,:,0][1], dp_exp_processed_1[0,:,0][2]])
org_2 = np.array([dp_exp_processed_1[0,:,1][0], dp_exp_processed_1[0,:,1][1], dp_exp_processed_1[0,:,1][2]])
org_3 = np.array([dp_exp_processed_1[0,:,2][0], dp_exp_processed_1[0,:,2][1], dp_exp_processed_1[0,:,2][2]])

vo1 = org_1 - org_2
vo2 = org_3 - org_2

head_v_org = np.cross( vo1, vo2)
a, b, c = head_v_org
d = np.dot(head_v_org, org_3 )

end_1 = np.array([dp_exp_processed_1[0,:,6][0], dp_exp_processed_1[0,:,6][1], dp_exp_processed_1[0,:,6][2]])
end_2 = np.array([dp_exp_processed_1[0,:,7][0], dp_exp_processed_1[0,:,7][1], dp_exp_processed_1[0,:,7][2]])
end_3 = np.array([dp_exp_processed_1[0,:,8][0], dp_exp_processed_1[0,:,8][1], dp_exp_processed_1[0,:,8][2]])

dis1 = np.abs(a*end_1[0] + b*end_1[1] + c*end_1[2] - d) / np.sqrt(a**2+b**2+c**2)
dis2 = np.abs(a*end_2[0] + b*end_2[1] + c*end_2[2] - d) / np.sqrt(a**2+b**2+c**2)
dis3 = np.abs(a*end_3[0] + b*end_3[1] + c*end_3[2] - d) / np.sqrt(a**2+b**2+c**2)
dis0 = np.abs(a*org_1[0] + b*org_1[1] + c*org_1[2] - d) / np.sqrt(a**2+b**2+c**2)









a = 1
exit()

print("end")











############ point  ###########
head_v_org = vec(0,0,1)

v1_org =vec(-0.36, 0, 0)
v2_org =vec(0.183, -0.317, 0)
v3_org =vec(0.183, 0.317, 0)

len_array_module1_act1_inverse = np.zeros([3])

current_1 = vec(-0.9309, -0.9784, 1.748)
current_2 = vec(-0.5436, -0.9416, 2.2485)
current_3 = vec(-0.3819, -1.2953, 1.748)
current_o = (current_1+current_2+current_3)/3

v1 = current_1 - current_o
v2 = current_2 - current_o
v3 = current_3 - current_o

v12 = current_1 - current_2
v32 = current_3 - current_2

head_v = vector.cross( v12.hat, v32.hat)

v1_Ang_xy_plane_to_X = Cal_two_vector_angle_2pi(v1_org.hat, vec(1,0,0))
v2_Ang_xy_plane_to_X = Cal_two_vector_angle_2pi(v2_org.hat, vec(1,0,0))
v3_Ang_xy_plane_to_X = Cal_two_vector_angle_2pi(v3_org.hat, vec(1,0,0))



#a.dot( b)/(np.linalg.norm(a)*np.linalg.norm(b))
head_v_xy_plane = vec(head_v.x, head_v.y, 0).hat
head_Ang_xy_plane_to_X = np.arccos(vector.dot(head_v_xy_plane, vec(1,0,0)))
sgn = vector.cross( vec(1,0,0), head_v_xy_plane).z #anticlock wise is +

head_Ang_xy_plane_to_X = Cal_two_vector_angle_2pi(head_v_xy_plane, vec(1,0,0) )#np.mod(head_Ang_xy_plane_to_X * (-1)*(sgn < 0), 2 * pi)

theta = np.arccos(vector.dot(head_v, head_v_org))

r_softArm = (6+1+0.2*2)/20
dis = (current_o - vec(0,0,0)).mag
R_center = (dis/2)/np.sin(theta/2)
print("theta={}, head_Ang_xy_plane_to_X={} ".format(np.rad2deg(theta),
                                   np.rad2deg(head_Ang_xy_plane_to_X)))

len_array_module1_act1_inverse[0] = (R_center - r_softArm*np.cos( v1_Ang_xy_plane_to_X + head_Ang_xy_plane_to_X )) * theta
len_array_module1_act1_inverse[1] = (R_center - r_softArm*np.cos( v2_Ang_xy_plane_to_X + head_Ang_xy_plane_to_X )) * theta
len_array_module1_act1_inverse[2] = (R_center - r_softArm*np.cos( v3_Ang_xy_plane_to_X + head_Ang_xy_plane_to_X )) * theta







len_array_module1_act1_inverse = np.zeros([3])

current_1 = vec(-0.9309, -0.9784, 1.748)
current_2 = vec(-0.5436, -0.9416, 2.2485)
current_3 = vec(-0.3819, -1.2953, 1.748)
current_o = (current_1+current_2+current_3)/3

v1 = current_1 - current_o
v2 = current_2 - current_o
v3 = current_3 - current_o

v12 = current_1 - current_2
v32 = current_3 - current_2

head_v = vector.cross( v12.hat, v32.hat)

v1_Ang_xy_plane_to_X = Cal_two_vector_angle_2pi(v1_org.hat, vec(1,0,0))
v2_Ang_xy_plane_to_X = Cal_two_vector_angle_2pi(v2_org.hat, vec(1,0,0))
v3_Ang_xy_plane_to_X = Cal_two_vector_angle_2pi(v3_org.hat, vec(1,0,0))



#a.dot( b)/(np.linalg.norm(a)*np.linalg.norm(b))
head_v_xy_plane = vec(head_v.x, head_v.y, 0).hat
head_Ang_xy_plane_to_X = np.arccos(vector.dot(head_v_xy_plane, vec(1,0,0)))
sgn = vector.cross( vec(1,0,0), head_v_xy_plane).z #anticlock wise is +

head_Ang_xy_plane_to_X = Cal_two_vector_angle_2pi(head_v_xy_plane, vec(1,0,0) )#np.mod(head_Ang_xy_plane_to_X * (-1)*(sgn < 0), 2 * pi)

theta = np.arccos(vector.dot(head_v, head_v_org))

r_softArm = (6+1+0.2*2)/20
dis = (current_o - vec(0,0,0)).mag
R_center = (dis/2)/np.sin(theta/2)
print("theta={}, head_Ang_xy_plane_to_X={} ".format(np.rad2deg(theta),
                                   np.rad2deg(head_Ang_xy_plane_to_X)))

len_array_module1_act1_inverse[0] = (R_center - r_softArm*np.cos( v1_Ang_xy_plane_to_X + head_Ang_xy_plane_to_X )) * theta
len_array_module1_act1_inverse[1] = (R_center - r_softArm*np.cos( v2_Ang_xy_plane_to_X + head_Ang_xy_plane_to_X )) * theta
len_array_module1_act1_inverse[2] = (R_center - r_softArm*np.cos( v3_Ang_xy_plane_to_X + head_Ang_xy_plane_to_X )) * theta



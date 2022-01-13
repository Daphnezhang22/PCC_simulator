from vpython import *
import numpy as np
import sys
import growbot
from growbot import *
import matplotlib.pyplot as plt
import time
import pandas as pd

scene2 = canvas(title='Simulation System', width=800, height=700, center=vector(0, 0, 0),
                background=color.white)

bottom_plane = box(pos=vec(0, 0, -0.1), axis=vec(1, 0, 0), length=2000, height=0.1,
                   width=2000, color=color.white, up=vec(0, 0, 1), opacity=1)  # color = vec(180/255,180/255,180/255)

#######   Camera & Light    ########
scene2.camera.up = vec(0, 0, 1)
x_camera_pos = vec(7.10204, -0.240105, 2.31849)
y_camera_pos = vec(-0.175975, 6.9243, 2.80983)
z_camera_pos = vec(-0.00125219, 0.563198, 7.45351)
sample_pos = vec(0.35646, 4.98352, 0.377328)
scene2.camera.pos = sample_pos  # (-6.60371, 1.34283, 2.26781) #(-4.21826, -6.77872, 2.1207)
scene2.camera.axis = sample_pos * -1  # (4.21826, 6.77872, -2.1207)
alterable_grow.LightUp4()

alterable_grow.Show_coordinator_xyz(-1, -1, 0.1)

#######   plant initial    ##########
# R0      = np.array([0.366, 0.266, 0.266, 0.266, 0.266]) #module radius
R0 = np.array([0.4, 0.3, 0.3, 0.3, 0.3])
spr_len = np.array([1, 1, 1, 1, 1])
rot_ang = np.array([0, pi / 3, 0, 0, 0])  # rad unit anti-clockwise
spr_dis = R0 * np.sqrt(3)
module_num = 10
test = alterable_grow.Cell(Module_num=module_num, Spr_distance=spr_dis,
                           Spr_len=spr_len, Shift_angle=rot_ang)
test.increase_all_sides(0.2)
t = 0
delt = 0.1
dt = spr_len * delt * 0.1
dt_L = spr_len * delt
rate(2)
test.spr_coils = 3
ini_len = 1

test.add_one_module_on_top(1)
test.add_one_module_on_top(0.01)
test.change_inner_module_lens(1, 0.6, 0.6, 0.6)
# test.change_inner_module_lens(1, 0.54, 0.47, 0.53)
test.add_one_module_on_top(1)

top_center_point = sphere(pos=test.get_top_center_positions(), radius=0.01,
                          color=color.white,
                          make_trail=True, opacity=0)

first_mod_center_point = sphere(pos=test.get_center_positions(10)[0], radius=0.01,
                                color=color.white,
                                make_trail=True, opacity=0)
sleep(2)
M1pos_recording_array = growbot.marker.Vpoint()
M2pos_recording_array = growbot.marker.Vpoint()
L1 = 0.3
L2 = 0.3
L3 = 0.3
ph1 = 0
ph2 = (2 * pi / 3)  # (1 * pi / 4) #
ph3 = (4 * pi / 3)  # (1 * pi / 2) #
all_end_position_module = []
ini_len = 1.5
a = 1
gravity_vec = vector(0, 0, 1)

################### cir       ######################
input_data = pd.read_excel('D:/Santanna/3D Grow/Trial_5/len_array_module1_act1_inverse.xls', header=None)
input_data = input_data / 10
m, n = input_data.shape
N = m
s1 = np.linspace(0, 2 * pi, N)
s2 = np.linspace(0, 2 * pi, N)
for j in range(1):
    L1 = 0.4  # (a + i)/8
    L2 = L1
    # print("L1 = L2 = {}".format(L1))
    for i in range(N):
        # print("i = {}, s1[i] = {}".format(i, s1[i]))
        '''
        len11 = L1 * np.sin((s1[i] + ph1)) + L1 * np.sin((s2[i] + ph1))  + ini_len
        len12 = L1 * np.sin((s1[i] + ph2)) + L1 * np.sin((s2[i] + ph2))  + ini_len
        len13 = L1 * np.sin((s1[i] + ph3)) + L1 * np.sin((s2[i] + ph3))  + ini_len
        #print("in layer: {0}, i = {1}, lens are {2}, {3},  {4} ".format(circ_num,i,len1, len2, len3 ))
        len21 = L2 * np.sin((s1[i] + ph1)) + L2 * np.sin((s2[i] + ph1))  + ini_len
        len22 = L2 * np.sin((s1[i] + ph2)) + L2 * np.sin((s2[i] + ph2))  + ini_len
        len23 = L2 * np.sin((s1[i] + ph3)) + L2 * np.sin((s2[i] + ph3))  + ini_len      

        len11 = L1 * np.sin((s1[i] + ph1)) + ini_len
        len12 = L1 * np.sin((s1[i] + ph2)) + ini_len
        len13 = L1 * np.sin((s1[i] + ph3)) + ini_len
        #print("in layer: {0}, i = {1}, lens are {2}, {3},  {4} ".format(circ_num,i,len1, len2, len3 ))
        len21 = L2 * np.sin((s2[i] + ph1 )) + ini_len
        len22 = L2 * np.sin((s2[i] + ph2 )) + ini_len
        len23 = L2 * np.sin((s2[i] + ph3 )) + ini_len
        '''
        '''
        len11 = pad_mod1[int((L1 * (np.sin((s1[i] + ph1)) + 1))*100)]  + ini_len_1
        len12 = pad_mod1[int((L1 * (np.sin((s1[i] + ph2)) + 1))*100)]  + ini_len_1
        len13 = pad_mod1[int((L1 * (np.sin((s1[i] + ph3)) + 1))*100)] + ini_len_1
        #print("in layer: {0}, i = {1}, lens are {2}, {3},  {4} ".format(circ_num,i,len1, len2, len3 ))
        len21 = pad_mod2[int((L2 * (np.sin((s2[i] + ph1 )) + 1))*100)]  + ini_len_2
        len22 = pad_mod2[int((L2 * (np.sin((s2[i] + ph2 )) + 1))*100)] + ini_len_2
        len23 = pad_mod2[int((L2 * (np.sin((s2[i] + ph3 )) + 1))*100)]  + ini_len_2       

        #print("in layer: {0}, i = {1}, lens are {2}, {3},  {4} ".format(circ_num,i,len1, len2, len3 ))
        #len31 = L3 * np.sin((s1[i] + ph1 )) + ini_len
        #len32 = L3 * np.sin((s1[i] + ph2 )) + ini_len
        #len33 = L3 * np.sin((s1[i] + ph3 )) + ini_len

        len11 = pad_mod2[int(( input_data[0][i] ))] + ini_len_1
        len12 = pad_mod2[int(( input_data[1][i] ))] + ini_len_1
        len13 = pad_mod2[int(( input_data[2][i] ))] + ini_len_1
        #print("in layer: {0}, i = {1}, lens are {2}, {3},  {4} ".format(circ_num,i,len1, len2, len3 ))
        len21 = pad_mod2[int(( input_data[3][i] ))] + ini_len_2
        len22 = pad_mod2[int(( input_data[4][i] ))] + ini_len_2
        len23 = pad_mod2[int(( input_data[5][i] ))] + ini_len_2
        '''
        len12 = input_data[0][i]
        len11 = input_data[1][i]
        len13 = input_data[2][i]
        print(f"i = {i}/{N}, lens are {len11}, {len12},  {len13} ")
        len22 = input_data[3][i]
        len21 = input_data[4][i]
        len23 = input_data[5][i]

        test.change_inner_module_lens(0, len12, len11, len13)
        test.change_inner_module_lens(2, len22, len21, len23)
        top_center_point.pos = test.get_top_center_positions()
        first_mod_center_point.pos = test.get_center_positions(10)[0]

        M2pos_recording_array.Center_Append(np.array(test.get_center_positions(30)[0].value))
        M1pos_recording_array.Center_Append(np.array(test.get_center_positions(10)[0].value))

'''       
df = pd.DataFrame(list(zip(Center_posX, Center_posY, Center_posZ, 
                            First_cen_posX, First_cen_posY, First_cen_posZ )), 
                    columns=["T_Center_posX","T_Center_posY", "T_Center_posZ", 
                            "M_Center_posX","M_Center_posY", "M_Center_posY"])
'''
df = pd.DataFrame(list(zip(M2pos_recording_array.Get_center_colomm(0),
                           M2pos_recording_array.Get_center_colomm(1),
                           M2pos_recording_array.Get_center_colomm(2),
                           M1pos_recording_array.Get_center_colomm(0),
                           M1pos_recording_array.Get_center_colomm(1),
                           M1pos_recording_array.Get_center_colomm(2), )),
                  columns=["T_Center_posX", "T_Center_posY", "T_Center_posZ",
                           "M_Center_posX", "M_Center_posY", "M_Center_posY"])

df.to_csv('Flower_full_2_36_change12_45.csv', index=False)
df.to_excel('Flower_full_2_36_change12_45.xls', index=False, header=False)
print("cir finished")
sleep(10000)
exit()

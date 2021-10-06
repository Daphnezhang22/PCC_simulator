from vpython import *
import numpy as np
import sys

sys.path.append(".")
import alterable_grow
from alterable_grow import *
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
LightUp4()

Show_coordinator_xyz(-1, -1, 0.1)

#######   plant initial    ##########
R0 = np.array([0.64, 0.45, 0.45, 0.45, 0.45])  # module diameter
rot_ang = np.array([0, pi / 3, 0, 0, 0])  # rad unit anti-clockwise
spr_len = np.array([1, 1, 1, 1, 1])
spr_dis = R0 * np.sqrt(3)
module_num = 10
test = alterable_grow.Cell(Module_num=module_num, Spr_distance=spr_dis, Spr_len=spr_len, Shift_angle=rot_ang)
test.increase_all_sides(0.2)
t = 0
delt = 0.1
dt = spr_len * delt * 0.1
dt_L = spr_len * delt
rate(2)
test.spr_coils = 3
ini_len = 1

test.add_one_module_on_top(1)
test.add_one_module_on_top(0.1)
# test.increasing_one_side_by_module(0.1, 0, 1)
# test.increasing_one_side_by_module(0.1, 1, 1)
test.add_one_module_on_top(1)

top_center_point = sphere(pos=test.get_top_center_positions(), radius=0.01,
                          color=color.white,
                          make_trail=True, opacity=0)

first_mod_center_point = sphere(pos=test.get_center_positions(10)[0], radius=0.01,
                                color=color.white,
                                make_trail=True, opacity=0)
sleep(2)
N = 20
# s1 = np.linspace(0,2*pi*10, N)
# s2 = np.linspace(0,2*pi*12, N)

s1 = np.linspace(0, 2 * pi, N)
s2 = np.linspace(0, 2 * pi, N)
P = 5
P1 = np.linspace(0, 1 * pi, P)
L1 = 0.3
L2 = 0.3
L3 = 0.3
ph1 = 0
ph2 = (2 * pi / 3)  # (1 * pi / 4) #
ph3 = (4 * pi / 3)  # (1 * pi / 2) #
Center_pos = []
Center_posX = []
Center_posY = []
Center_posZ = []
Center_posX.append(0)
Center_posY.append(0)
Center_posZ.append(0)
First_cen_posX = []
First_cen_posY = []
First_cen_posZ = []
First_cen_posX.append(0)
First_cen_posY.append(0)
First_cen_posZ.append(0)
ini_len = 1.5
a = 1
gravity_vec = vector(0, 0, 1)

###############      3 lines        ##################
ini_len_1 = 2
ini_len_2 = 1.4
N = 21

L_increase1 = np.linspace(ini_len_1, ini_len_1 + 0.64, N)
L_increase2 = np.linspace(ini_len_2, ini_len_2 + 1.1, N)

len_cali_module1 = np.array([[24.35767972, 26.97124734, 26.98213086],
                             [24.35804709, 26.96699103, 26.97968648],
                             [24.35781464, 26.96639599, 26.97762132],
                             [24.35590834, 26.96698949, 26.97956355],
                             [24.36705679, 26.96471771, 27.00822047],
                             [24.37880418, 26.97047292, 27.02242363],
                             [24.41716108, 26.99717551, 27.03037254],
                             [24.47048114, 27.01188621, 27.08023284],
                             [24.53550398, 27.04518773, 27.20486088],
                             [24.61801863, 27.10286879, 27.23923038],
                             [24.71318571, 27.15045995, 27.27070655],
                             [24.79973974, 27.21739036, 27.30576271],
                             [24.91524477, 27.28087275, 27.42163046],
                             [25.02862509, 27.35584599, 27.49927925],
                             [25.15303217, 27.49573919, 27.59262993],
                             [25.30614388, 27.62240758, 27.67740448],
                             [25.44696034, 27.72581949, 27.7847625],
                             [25.52685918, 27.83236477, 27.84453552],
                             [25.68507367, 27.94308202, 27.95359092],
                             [25.81756181, 28.08324421, 28.0632716],
                             [26.02539918, 28.30864776, 28.20806681],
                             [26.29516769, 28.59646566, 28.42265632],
                             [26.54239792, 28.80716077, 28.57940917],
                             [26.6937438, 28.97450833, 28.76427788],
                             [26.91385455, 29.13406833, 28.99502357],
                             [27.14739655, 29.37137356, 29.0494076],
                             [27.34753468, 29.58251591, 29.34091517],
                             [27.58724765, 29.80211964, 29.53061808],
                             [27.81248236, 30.03129118, 29.68188392],
                             [27.97476165, 30.23388789, 29.81866228],
                             [28.12782033, 30.36645694, 29.97362794],
                             [28.36156987, 30.59789411, 30.17911234],
                             [28.61365928, 30.87321354, 30.39437752],
                             [28.8610107, 31.16328594, 30.59971083],
                             [29.19520767, 31.52632255, 30.86972534],
                             [29.39159734, 31.77793366, 31.02172602],
                             [29.60264801, 32.00243847, 31.2242306],
                             [29.84669871, 32.27967804, 31.46848017],
                             [30.11027742, 32.59520101, 31.7220006],
                             [30.37009246, 32.88785686, 31.99811326],
                             [30.63395527, 33.21430343, 32.24942101]]) * 0.1 - 0.4

len_cali_module2 = np.array([[16.36019962, 16.08924319, 16.35236972],
                             [16.36461613, 16.09050283, 16.34630899],
                             [16.36317516, 16.09276863, 16.34064042],
                             [16.36182183, 16.08891067, 16.3475796],
                             [16.36868278, 16.09980937, 16.32238174],
                             [16.37556735, 16.10630673, 16.36071782],
                             [16.40362776, 16.13444192, 16.36344068],
                             [16.43060444, 16.16119098, 16.39144716],
                             [16.46477573, 16.19882956, 16.42717356],
                             [16.52646725, 16.25176832, 16.47203882],
                             [16.5740366, 16.30982664, 16.5154135],
                             [16.58714131, 16.38929368, 16.59552356],
                             [16.6236136, 16.47400269, 16.62900199],
                             [16.70583213, 16.53793855, 16.73044826],
                             [16.77752559, 16.61145153, 16.8027826],
                             [16.86892332, 16.69668947, 16.87353319],
                             [16.99302649, 16.769715, 16.94338335],
                             [17.08037703, 16.81038567, 17.01574804],
                             [17.17606803, 16.89675518, 17.12362189],
                             [17.28851117, 16.99564708, 17.18997733],
                             [17.40155536, 17.0947696, 17.31378735],
                             [17.59054502, 17.23663283, 17.47271769],
                             [17.71443521, 17.37371, 17.62318817],
                             [17.84274821, 17.41718706, 17.69390108],
                             [17.99059829, 17.52619047, 17.82632595],
                             [18.18587865, 17.67361399, 18.01138444],
                             [18.33983839, 17.7966872, 18.14701394],
                             [18.5303186, 17.91850849, 18.35206631],
                             [18.7306478, 18.082207, 18.52477872],
                             [18.81862962, 18.2165816, 18.65717168],
                             [18.96294553, 18.32040277, 18.79024362],
                             [19.17520276, 18.49584732, 18.98290092],
                             [19.39862741, 18.66803845, 19.17506441],
                             [19.63852326, 18.82723612, 19.41341994],
                             [19.95853216, 19.06545917, 19.7213607],
                             [20.11523402, 19.23582353, 19.8045872],
                             [20.36499071, 19.41383549, 20.08725454],
                             [20.62613845, 19.59624961, 20.36054507],
                             [20.89252663, 19.79170563, 20.57911293],
                             [21.17828053, 20.05171069, 20.81330991],
                             [21.46022072, 20.27171749, 21.10478202]]) * 0.1

Mod1_sample41 = np.array([0.00000000e+00, 4.02010537e-04, 2.51746603e-03, -1.75891664e-03,
                          1.62328294e-03, -2.16534424e-02, -4.85967289e-02, -8.57139905e-02,
                          -1.35534876e-01, -2.00847305e-01, -2.50544195e-01, -3.25500602e-01,
                          -3.86156498e-01, -4.71752802e-01, -5.44060269e-01, -6.26533274e-01,
                          -7.02854444e-01, -7.68675761e-01, -8.67041477e-01, -9.54241475e-01,
                          -1.05918589e+00, -1.20750767e+00, -1.31825404e+00, -1.38403732e+00,
                          -1.51096311e+00, -1.68516275e+00, -1.81065206e+00, -1.97585917e+00,
                          -2.13353228e+00, -2.23771988e+00, -2.36589979e+00, -2.55069788e+00,
                          -2.72988332e+00, -2.90951463e+00, -3.17910780e+00, -3.27021214e+00,
                          -3.49667651e+00, -3.72364374e+00, -3.92460634e+00, -4.16643683e+00,
                          -4.39840817e+00]) * -0.1

Mod2_sample41 = np.array([0.00000000e+00, 2.28574761e-03, 3.42082108e-03, 3.35691375e-03,
                          -1.17198621e-02, -2.22536807e-02, -4.14602904e-02, -7.72265363e-02,
                          -1.45233731e-01, -1.97646212e-01, -2.49375765e-01, -3.08237402e-01,
                          -3.98853480e-01, -4.82648236e-01, -5.97426425e-01, -7.12801033e-01,
                          -8.24374689e-01, -9.13919766e-01, -1.03860933e+00, -1.17521099e+00,
                          -1.36909474e+00, -1.63431993e+00, -1.83953971e+00, -2.01504842e+00,
                          -2.21542195e+00, -2.39121509e+00, -2.63325338e+00, -2.85227012e+00,
                          -3.05238010e+00, -3.22409564e+00, -3.37680595e+00, -3.60685092e+00,
                          -3.86126037e+00, -4.11388172e+00, -4.44227642e+00, -4.64133289e+00,
                          -4.85815939e+00, -5.11469216e+00, -5.39669490e+00, -5.68000061e+00,
                          -5.96142753e+00]) * -0.1

L_increase1_Z = np.array([0.00000000e+00, -1.22891053e-02, -3.81927319e-02, -1.00101228e-01,
                          -6.93410740e-01, -1.84634396e+00, -2.97366545e+00, -4.51763125e+00,
                          -6.24701194e+00, -7.60785767e+00, -9.32499296e+00, -1.18950081e+01,
                          -1.37928414e+01, -1.67108070e+01, -1.98530416e+01, -2.23497914e+01,
                          -2.55250453e+01, -2.92048481e+01, -3.29773577e+01, -3.70131400e+01,
                          -4.17121724e+01]) * -0.01

L_increase2_Z = np.array([0.00000000e+00, -4.27844585e-03, -1.04385034e-03, -3.46098306e-01,
                          -9.24966129e-01, -2.03502427e+00, -3.19216647e+00, -4.79940457e+00,
                          -6.94662234e+00, -8.89652817e+00, -1.14686560e+01, -1.62425465e+01,
                          -1.98427941e+01, -2.37151817e+01, -2.81797382e+01, -3.20686622e+01,
                          -3.58990290e+01, -4.10911306e+01, -4.66806913e+01, -5.16469765e+01,
                          -5.73610884e+01]) * -0.01

pressure = np.linspace(0, 0.8, 81)
pad_mod1 = []
pad_mod2 = []
for i in range(Mod2_sample41.size * 2 - 1):
    if i % 2 != 0:
        avg1 = (Mod1_sample41[int(i / 2)] + Mod1_sample41[int(i / 2 + 1)]) / 2
        avg2 = (Mod2_sample41[int(i / 2)] + Mod2_sample41[int(i / 2 + 1)]) / 2
        pad_mod1.append(avg1)
        pad_mod2.append(avg2)
    else:
        pad_mod1.append(Mod1_sample41[int((i + 1) / 2)])
        pad_mod2.append(Mod2_sample41[int((i + 1) / 2)])

ini_len_1 = 2
ini_len_2 = 1.5
N = 21

len_arr1 = np.ones((3, N, 3)) * ini_len_1  # np.ones((3,N,3))*ini_len_1
len_arr2 = np.ones((3, N, 3)) * ini_len_2  # np.ones((3,N,3))*ini_len_2

len_arr1[0, :, 0] = L_increase1_Z + ini_len_1
len_arr1[1, :, 1] = L_increase1_Z + ini_len_1
len_arr1[2, :, 2] = L_increase1_Z + ini_len_1

len_arr2[0, :, 0] = L_increase2_Z + ini_len_2
len_arr2[1, :, 1] = L_increase2_Z + ini_len_2
len_arr2[2, :, 2] = L_increase2_Z + ini_len_2
L = np.zeros(3)

################### cir       ######################
input_data = pd.read_excel('/home/qian/input_data.xls', header=None)

N = 51
s1 = np.linspace(0, 2 * pi, N)
s2 = np.linspace(0, 2 * pi, N)
for j in range(1):
    L1 = 0.4  # (a + i)/8
    L2 = L1
    print("L1 = L2 = {}".format(L1))
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
        '''
        # print("in layer: {0}, i = {1}, lens are {2}, {3},  {4} ".format(circ_num,i,len1, len2, len3 ))
        # len31 = L3 * np.sin((s1[i] + ph1 )) + ini_len
        # len32 = L3 * np.sin((s1[i] + ph2 )) + ini_len
        # len33 = L3 * np.sin((s1[i] + ph3 )) + ini_len

        len11 = pad_mod2[int((input_data[0][i]) * 100)] + ini_len_1
        len12 = pad_mod2[int((input_data[1][i]) * 100)] + ini_len_1
        len13 = pad_mod2[int((input_data[2][i]) * 100)] + ini_len_1
        # print("in layer: {0}, i = {1}, lens are {2}, {3},  {4} ".format(circ_num,i,len1, len2, len3 ))
        len21 = pad_mod2[int((input_data[3][i]) * 100)] + ini_len_2
        len22 = pad_mod2[int((input_data[4][i]) * 100)] + ini_len_2
        len23 = pad_mod2[int((input_data[5][i]) * 100)] + ini_len_2

        test.change_inner_module_lens(0, len12, len11, len13)
        test.change_inner_module_lens(2, len22, len21, len23)
        top_center_point.pos = test.get_top_center_positions()
        First_module_center_point = test.get_center_positions(10)[0]
        first_mod_center_point.pos = test.get_center_positions(10)[0]
        Center_posX.append(top_center_point.pos.x)
        Center_posY.append(top_center_point.pos.y)
        Center_posZ.append(top_center_point.pos.z)
        First_cen_posX.append(First_module_center_point.x)
        First_cen_posY.append(First_module_center_point.y)
        First_cen_posZ.append(First_module_center_point.z)

    test.change_inner_module_lens(1, ini_len, ini_len, ini_len)

    Center_posX.append(0)
    Center_posY.append(0)
    Center_posZ.append(0)
    First_cen_posX.append(0)
    First_cen_posY.append(0)
    First_cen_posZ.append(0)

df = pd.DataFrame(list(zip(Center_posX, Center_posY, Center_posZ,
                           First_cen_posX, First_cen_posY, First_cen_posZ)),
                  columns=["T_Center_posX", "T_Center_posY", "T_Center_posZ",
                           "M_Center_posX", "M_Center_posY", "M_Center_posY"])
df.to_csv('cir_module_spiral_piby2.csv', index=False)
print("cir finished")
exit()

###################### 1 by 1
N = int(len_cali_module1.size / 3)
for j in range(1):

    for i in range(N):
        print("i={}".format(i))
        '''
        len11 = len_arr1[0, i, j] 
        len12 = len_arr1[1, i, j] 
        len13 = len_arr1[2, i, j]  

        len21 = len_arr2[0, i, j] 
        len22 = len_arr2[1, i, j] 
        len23 = len_arr2[2, i, j]   
        '''

        len11 = len_cali_module1[i, 0]
        len12 = len_cali_module1[i, 2]
        len13 = len_cali_module1[i, 1]

        len21 = len_cali_module2[i, 1]
        len22 = len_cali_module2[i, 2]
        len23 = len_cali_module2[i, 1]
        '''
        len11 = len_array_module1[i,0]
        len12 = len_array_module1[i,1]
        len13 = len_array_module1[i,2]  

        len21 = len_array_module2[i,0] 
        len22 = len_array_module2[i,1] 
        len23 = len_array_module2[i,2]
        '''
        print("len11: {}, len12: {}, len13: {}".format(len11, len12, len13))
        print("len21: {}, len22: {}, len23: {}".format(len21, len22, len23))

        test.change_inner_module_lens(0, len11, len12, len13)
        test.change_inner_module_lens(2, len21, len22, len23)

        top_center_point.pos = test.get_top_center_positions()
        First_module_center_point = test.get_center_positions(10)[0]
        first_mod_center_point.pos = test.get_center_positions(10)[0]
        Center_posX.append(top_center_point.pos.x)
        Center_posY.append(top_center_point.pos.y)
        Center_posZ.append(top_center_point.pos.z)
        First_cen_posX.append(First_module_center_point.x)
        First_cen_posY.append(First_module_center_point.y)
        First_cen_posZ.append(First_module_center_point.z)
    Center_posX.append(0)
    Center_posY.append(0)
    Center_posZ.append(0)
    First_cen_posX.append(0)
    First_cen_posY.append(0)
    First_cen_posZ.append(0)

df = pd.DataFrame(list(zip(Center_posX, Center_posY, Center_posZ,
                           First_cen_posX, First_cen_posY, First_cen_posZ)),
                  columns=["T_Center_posX", "T_Center_posY", "T_Center_posZ",
                           "M_Center_posX", "M_Center_posY", "M_Center_posY"])
df.to_csv('lines_test_full_single_data_cal_132.csv', index=False)
print("data finished")
exit()

#################### 3 lines max ############
for j in range(1):
    print("in for loop 1_46")
    for i in range(N):
        j = 0
        len11 = len_arr1[0, i, j]
        len12 = len_arr1[1, i, j]
        len13 = len_arr1[2, i, j]
        len21 = len_arr2[0, i, j]
        len22 = len_arr2[1, i, j]
        len23 = len_arr2[2, i, j]
        test.change_inner_module_lens(0, len11, len12, len13)
        test.change_inner_module_lens(2, len21, len21, len23)
        top_center_point.pos = test.get_top_center_positions()
        First_module_center_point = test.get_center_positions(10)[0]
        Center_posX.append(top_center_point.pos.x)
        Center_posY.append(top_center_point.pos.y)
        Center_posZ.append(top_center_point.pos.z)
        First_cen_posX.append(First_module_center_point.x)
        First_cen_posY.append(First_module_center_point.y)
        First_cen_posZ.append(First_module_center_point.z)
    Center_posX.append(0)
    Center_posY.append(0)
    Center_posZ.append(0)
    First_cen_posX.append(0)
    First_cen_posY.append(0)
    First_cen_posZ.append(0)

for j in range(1):
    print("in for loop 2_45")
    for i in range(N):
        j = 1
        len11 = len_arr1[0, i, j]
        len12 = len_arr1[1, i, j]
        len13 = len_arr1[2, i, j]
        len21 = len_arr2[0, i, j]
        len22 = len_arr2[1, i, j]
        len23 = len_arr2[2, i, j]
        test.change_inner_module_lens(0, len11, len12, len13)
        test.change_inner_module_lens(2, len21, len22, len22)
        top_center_point.pos = test.get_top_center_positions()
        First_module_center_point = test.get_center_positions(10)[0]
        Center_posX.append(top_center_point.pos.x)
        Center_posY.append(top_center_point.pos.y)
        Center_posZ.append(top_center_point.pos.z)
        First_cen_posX.append(First_module_center_point.x)
        First_cen_posY.append(First_module_center_point.y)
        First_cen_posZ.append(First_module_center_point.z)
    Center_posX.append(0)
    Center_posY.append(0)
    Center_posZ.append(0)
    First_cen_posX.append(0)
    First_cen_posY.append(0)
    First_cen_posZ.append(0)

for j in range(1):
    print("in for loop 3_56")
    for i in range(N):
        j = 2
        len11 = len_arr1[0, i, j]
        len12 = len_arr1[1, i, j]
        len13 = len_arr1[2, i, j]
        len21 = len_arr2[0, i, j]
        len22 = len_arr2[1, i, j]
        len23 = len_arr2[2, i, j]
        test.change_inner_module_lens(0, len11, len12, len13)
        test.change_inner_module_lens(2, len23, len22, len23)
        top_center_point.pos = test.get_top_center_positions()
        First_module_center_point = test.get_center_positions(10)[0]
        Center_posX.append(top_center_point.pos.x)
        Center_posY.append(top_center_point.pos.y)
        Center_posZ.append(top_center_point.pos.z)
        First_cen_posX.append(First_module_center_point.x)
        First_cen_posY.append(First_module_center_point.y)
        First_cen_posZ.append(First_module_center_point.z)
    Center_posX.append(0)
    Center_posY.append(0)
    Center_posZ.append(0)
    First_cen_posX.append(0)
    First_cen_posY.append(0)
    First_cen_posZ.append(0)

df = pd.DataFrame(list(zip(Center_posX, Center_posY, Center_posZ,
                           First_cen_posX, First_cen_posY, First_cen_posZ)),
                  columns=["T_Center_posX", "T_Center_posY", "T_Center_posY",
                           "M_Center_posX", "M_Center_posY", "M_Center_posY"])
df.to_csv('lines_test_full_double_with_realZ.csv', index=False)
print("data finished")
exit()

'''  #gravity calculation 
        factor = test.get_module_impact_from_vector(0, gravity_vec)*0.1    
        print("factor = {}".format(factor))
        for k in range(3):                           
            L[k] = len_arr1[k, i, j] + (k if j ==k else 1)*factor


        for k in range(3):
            if L[j] < L.mean(): 
                L[k] = len_arr1[k, i, j]            

        len11 = len_arr1[0, i, j] + (0 if j ==0 else 1)*factor
        len12 = len_arr1[1, i, j] + (0 if j ==1 else 1)*factor
        len13 = len_arr1[2, i, j] + (0 if j ==2 else 1)*factor

        len21 = len_arr2[0, i, j] 
        len22 = len_arr2[1, i, j] 
        len23 = len_arr2[2, i, j] 

        print("len11: {}, len12: {}, len13: {}".format(len11, len12, len13))
        print("L1: {}, L2: {}, L3: {}".format(L[0], L[1], L[2]))
'''

'''
sleep(5)
test.increasing_one_side_by_module(0.2, 1, 0)
test.increasing_one_side_by_module(0.2, 1, 1)
test.increasing_one_side_by_module(0.2, 2, 2)
test.increasing_one_side_by_module(0.2, 2, 3)
test.increasing_one_side_by_module(0.2, 2, 4)
print("end")





L1 = 0.3
L2 = 0.3
ini_len = 1.2
for  i in range(N):
    len11 = L1 * np.sin((s1[i] + ph1)) + ini_len
    len12 = L1 * np.sin((s1[i] + ph2)) + ini_len
    len13 = L1 * np.sin((s1[i] + ph3)) + ini_len
    #print("in layer: {0}, i = {1}, lens are {2}, {3},  {4} ".format(circ_num,i,len1, len2, len3 ))
    len21 = L2 * np.sin((s1[i] + ph1 + pi/3)) + ini_len
    len22 = L2 * np.sin((s1[i] + ph2 + pi/3)) + ini_len
    len23 = L2 * np.sin((s1[i] + ph3 + pi/3)) + ini_len   
    test.change_inner_module_lens(0, len11, len12, len13)
    test.change_inner_module_lens(1, len21, len22, len23)
    top_center_point.pos = test.get_top_center_positions()
    Center_posX.append(top_center_point.pos.x)
    Center_posY.append(top_center_point.pos.y)
    Center_posZ.append(top_center_point.pos.z)

test.change_inner_module_lens(1, ini_len, ini_len, ini_len)

Center_posX.append(0)
Center_posY.append(0)
Center_posZ.append(0)





'''

L_increase1 = np.array([0.00000000e+00, -1.22891053e-02, -3.81927319e-02, -1.00101228e-01,
                        -6.93410740e-01, -1.84634396e+00, -2.97366545e+00, -4.51763125e+00,
                        -6.24701194e+00, -7.60785767e+00, -9.32499296e+00, -1.18950081e+01,
                        -1.37928414e+01, -1.67108070e+01, -1.98530416e+01, -2.23497914e+01,
                        -2.55250453e+01, -2.92048481e+01, -3.29773577e+01, -3.70131400e+01,
                        -4.17121724e+01]) * -0.01

L_increase2 = np.array([0.00000000e+00, -4.27844585e-03, -1.04385034e-03, -3.46098306e-01,
                        -9.24966129e-01, -2.03502427e+00, -3.19216647e+00, -4.79940457e+00,
                        -6.94662234e+00, -8.89652817e+00, -1.14686560e+01, -1.62425465e+01,
                        -1.98427941e+01, -2.37151817e+01, -2.81797382e+01, -3.20686622e+01,
                        -3.58990290e+01, -4.10911306e+01, -4.66806913e+01, -5.16469765e+01,
                        -5.73610884e+01]) * -0.01

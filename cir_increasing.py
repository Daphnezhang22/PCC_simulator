import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#book = xlrd.open_workbook("new_data_cir_lines.csv")
#dp_cir_exp   = pd.read_excel('exp_cir_2border_piby3_dif_order.xls', header=None )
dp_cir_exp   = pd.read_excel('D:/Santanna/3D Grow/Trial 3/Spiral/piby2/Spiral_piby2.xls', header=None )
dp_cir_simu   = pd.read_excel('D:/Santanna/3D Grow/Trial 3/Spiral/piby2/cir_module_spiral_piby2.xls', header=None )
#input_data   = pd.read_excel('D:/Santanna/3D Grow/Trial 3/input_data.xls', header=None )
#process raw data
dp_exp_processed_1 = np.zeros([int(dp_cir_exp[1].size/3),3, 2])
for i in range(int(dp_cir_exp[10].size/3)):
    j = 10
    dp_exp_processed_1[i, 0, 0] = dp_cir_exp[j][i*3]  # X
    dp_exp_processed_1[i, 1, 0] = dp_cir_exp[j][i*3+1]  # Y
    dp_exp_processed_1[i, 2, 0] = dp_cir_exp[j][i*3+2] # Z


dp_exp_processed_2 = np.zeros([int(dp_cir_exp[1].size/3),3, 2])
for i in range(int(dp_cir_exp[10].size/3)):
    j = 11
    dp_exp_processed_1[i, 0, 1] = dp_cir_exp[j][i*3]  # X
    dp_exp_processed_1[i, 1, 1] = dp_cir_exp[j][i*3+1]  # Y
    dp_exp_processed_1[i, 2, 1] = dp_cir_exp[j][i*3+2] # Z

dp_exp_processed_1 = dp_exp_processed_1/10
dp_exp_processed_2 = dp_exp_processed_2/10




dp_cir_simu[0]   = dp_cir_simu[0]*1*10
dp_cir_simu[1]   = dp_cir_simu[1]*1*10
dp_cir_simu[2]   = dp_cir_simu[2]*-1*10
dp_cir_simu[3]   = dp_cir_simu[3]*1*10
dp_cir_simu[4]   = dp_cir_simu[4]*1*10
dp_cir_simu[5]   = dp_cir_simu[5]*-1*10

theta = np.deg2rad(0)
#rotation X Y
for i in range(2):
    new_x = dp_cir_simu[0 + i * 3].mul(np.cos(theta)) - dp_cir_simu[1 + i * 3].mul( np.sin(theta) )
    new_y = dp_cir_simu[0 + i * 3].mul(np.sin(theta)) + dp_cir_simu[1 + i * 3].mul( np.cos(theta) )
    dp_cir_simu[0 + i * 3] = new_x
    dp_cir_simu[1 + i * 3] = new_y

'''
for i in range(1):
    dp_cir_simu[2+i*3] = dp_cir_simu[2+i*3]/-10 
    dp_cir_simu[0 + i * 3] = dp_cir_simu[0 + i * 3]
    dp_cir_simu[1 + i * 3] = dp_cir_simu[1 + i * 3]
'''

#position
for i in range(2):
    dp_cir_simu[0 + i * 3] = dp_cir_simu[0 + i * 3] - 1 # move to center
    dp_cir_simu[1 + i * 3] = dp_cir_simu[1 + i * 3] + 0.2
    dp_cir_simu[2 + i * 3] = dp_cir_simu[2 + i * 3] - 0

x_exp = np.concatenate(( dp_exp_processed_1[:, 0, 0] ,dp_exp_processed_1[:, 0, 1],

                         ) )

y_exp = np.concatenate(( dp_exp_processed_1[:, 1, 0] ,dp_exp_processed_1[:, 1, 1],


                         ) )

z_exp = np.concatenate(( dp_exp_processed_1[:, 2, 0] ,dp_exp_processed_1[:, 2, 1],

                         ) )


x_sim = pd.concat([dp_cir_simu[0], dp_cir_simu[3]])
y_sim = pd.concat([dp_cir_simu[1], dp_cir_simu[4]])
z_sim = pd.concat([dp_cir_simu[2], dp_cir_simu[5]])

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.axis('auto')
#ax.scatter(dp_lines[0], dp_lines[1], dp_lines[2], c='skyblue', s=60)
#ax.scatter(x, y, z, c='skyblue', s=10)
#ax.scatter(x_sim, y_sim, z_sim, c='turquoise', s=10)
#ax.scatter(x_exp, y_exp, z_exp, c='turquoise', label ='experiment')
#ax.scatter(x_sim, y_sim, z_sim, c='skyblue',  label = 'simulation')
ax.plot(x_sim, y_sim,z_sim, label = "Simulation")
ax.plot(x_exp, y_exp,z_exp, label = "Experiement ")



ax.legend()
plt.show()

a = 1





'''


'''
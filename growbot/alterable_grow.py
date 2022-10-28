from vpython import *
import numpy as np
# from scipy.spatial.transform import Rotation as R
import math
import sys
from sympy import symbols, solve, Eq, Function
import quaternion as quat
import time


class Cell:

    def __init__(self,Module_num, Spr_distance, Spr_len, **kwargs):
        self.module_num = Module_num
        self.spr_distance = np.append(Spr_distance[0], np.repeat(Spr_distance, Module_num) )   #array of layer number
        self.spr_len = np.append(Spr_len[0],  np.repeat(Spr_len, Module_num)    )          #array of layer number
        Shift_angle = kwargs.get('Shift_angle')
        Shift_angle_array = np.zeros(Module_num * len(Spr_distance))
        Shift_angle_array[0::Module_num] = Shift_angle
        self.shift_angle = np.append(0 ,Shift_angle_array)     #make size is same
        self.spr_len_thickness = 0.02
        self.unit_len_thickness = 0.005
        self.spr_coils = 3
        self.spr_radius = 0.08
        self.spr_color_vec = vec(144 / 255, 238 / 255, 144 / 255)
        self.spr_color_unit = vec(144 / 255, 238 / 255, 144 / 255)
        self.opacity = 0.6
        self.inner_vec_opacity = 0.3
        self.lateral_opacity = 0.5
        self.centeral_opacity = 0.9
        self.ini_len = Spr_distance[0] * 0.01  # ratio to be done
        self.max_len = Spr_distance[0] * 0.5  # 1.8                     #ratio to be done
        self.max_dif_len = Spr_distance[0]  # 2                  #ratio to be done


        self.walls = []  # MEMERY FOR POSITION AND LENGHT OF SPRINGS
        self.spring1 = []  # SPRING INFOR LIST
        self.spring2 = []
        self.spring3 = []
        self.spring_len = []
        self.springC = []
        self.Vec_12 = []
        self.Vec_23 = []
        self.Vec_31 = []
        self.top_cycle = []
        self.dist_springs2target = []
        x = 0
        y = 0
        x1 = x + self.spr_distance[0] / 2
        y1 = y - self.spr_distance[0] / 2 / np.sqrt(3)
        x2 = x - self.spr_distance[0] / 2
        y2 = y - self.spr_distance[0] / 2 / np.sqrt(3)
        x3 = x
        y3 = y + self.spr_distance[0] / 2 / np.sqrt(3) * 2

        self.walls.append(vector(x1, y1, 0))
        self.walls.append(vector(0, 0, self.ini_len))
        self.walls.append(vector(x2, y2, 0))
        self.walls.append(vector(0, 0, self.ini_len))
        self.walls.append(vector(x3, y3, 0))
        self.walls.append(vector(0, 0, self.ini_len))
        self.spring_len.append([self.ini_len, self.ini_len, self.ini_len])
        self.spring1[len(self.spring1):] = [cylinder(pos=self.walls[0],
                                                  axis=self.walls[1],
                                                  #thickness=self.spr_len_thickness, coils=self.spr_coils,
                                                  radius=self.spr_len_thickness,
                                                  color=self.spr_color_vec,
                                                  opacity=self.lateral_opacity)]

        self.spring2[len(self.spring2):] = [cylinder(pos=self.walls[2],
                                                  axis=self.walls[3],
                                                  #thickness=self.spr_len_thickness, coils=self.spr_coils,
                                                  radius=self.spr_len_thickness,
                                                  color=self.spr_color_vec,
                                                  opacity=self.lateral_opacity)]

        self.spring3[len(self.spring3):] = [cylinder(pos=self.walls[4],
                                                  axis=self.walls[5],
                                                  #thickness=self.spr_len_thickness, coils=self.spr_coils,
                                                  radius=self.spr_len_thickness,
                                                  color=self.spr_color_vec,
                                                  opacity=self.lateral_opacity)]

        # self.springC[len(self.springC):] = [helix(pos = ((self.walls[0] + self.walls[2] + self.walls[4] )/3),
        #                                         axis = ((self.walls[1] + self.walls[3] + self.walls[5] )/3),
        #                                         thickness = self.spr_len_thickness, coils = self.spr_coils,
        #                                         radius = self.spr_radius, color = vec(19/255, 153/255, 36/255) )]

        self.Vec_12[len(self.Vec_12):] = [cylinder(pos=self.walls[0] + self.walls[1],
                                                   axis=vector.norm(self.walls[2] + self.walls[3] - (
                                                           self.walls[0] + self.walls[1])) * self.spr_distance[0],
                                                   radius=self.unit_len_thickness, color=self.spr_color_vec,
                                                   thickness=self.unit_len_thickness,
                                                   opacity=0)]

        self.Vec_23[len(self.Vec_23):] = [cylinder(pos=self.walls[2] + self.walls[3],
                                                   axis=vector.norm(self.walls[4] + self.walls[5] - (
                                                           self.walls[2] + self.walls[3])) * self.spr_distance[0],
                                                   radius=self.unit_len_thickness, color=self.spr_color_vec,
                                                   thickness = self.unit_len_thickness,
                                                   opacity=0)]

        self.Vec_31[len(self.Vec_31):] = [cylinder(pos=self.walls[4] + self.walls[5],
                                                   axis=vector.norm(self.walls[0] + self.walls[1] - (
                                                           self.walls[4] + self.walls[5])) * self.spr_distance[0],
                                                   radius=self.unit_len_thickness, color=self.spr_color_vec,
                                                   thickness=self.unit_len_thickness,
                                                   opacity=0)]

        ring_rad = self.spr_distance[0] / np.sqrt(3)
        self.top_cycle[len(self.top_cycle):] = [ring(pos=(self.Vec_12[len(self.Vec_12) - 1].pos + self.Vec_23[
            len(self.Vec_23) - 1].pos + self.Vec_31[len(self.Vec_31) - 1].pos) / 3,
                                                     axis=vector.cross(self.Vec_23[len(self.Vec_23) - 1].axis,
                                                                       self.Vec_12[len(self.Vec_12) - 1].axis),
                                                     radius=ring_rad, thickness=self.unit_len_thickness,
                                                     color=vec(0 / 255, 102 / 255, 0 / 255), #self.spr_color_vec,
                                                     opacity=self.lateral_opacity)] #self.lateral_opacity)]



    def update_top_segment(self):
        # plot the top
        Num_layer = int(self.walls.__len__() / 6)
        self.spring1[len(self.spring1) - 1].axis = self.walls[(Num_layer - 1) * 6 + 1]
        self.spring2[len(self.spring2) - 1].axis = self.walls[(Num_layer - 1) * 6 + 3]
        self.spring3[len(self.spring3) - 1].axis = self.walls[(Num_layer - 1) * 6 + 5]
        # self.springC[len(self.spring3) - 1].axis = (self.walls[(Num_layer - 1) * 6 + 1] + self.walls[(Num_layer - 1) * 6 + 3]+ self.walls[(Num_layer - 1) * 6 + 5] )/3
        self.Vec_12[len(self.Vec_12) - 1].pos = self.walls[(Num_layer - 1) * 6 + 0] + self.walls[
            (Num_layer - 1) * 6 + 1]
        self.Vec_12[len(self.Vec_12) - 1].axis = vector.norm(
            self.walls[(Num_layer - 1) * 6 + 2] + self.walls[(Num_layer - 1) * 6 + 3] - (
                    self.walls[(Num_layer - 1) * 6 + 0] + self.walls[(Num_layer - 1) * 6 + 1])) * self.spr_distance[Num_layer - 1]
        self.Vec_23[len(self.Vec_23) - 1].pos = self.walls[(Num_layer - 1) * 6 + 2] + self.walls[
            (Num_layer - 1) * 6 + 3]
        self.Vec_23[len(self.Vec_23) - 1].axis = vector.norm(
            self.walls[(Num_layer - 1) * 6 + 4] + self.walls[(Num_layer - 1) * 6 + 5] - (
                    self.walls[(Num_layer - 1) * 6 + 2] + self.walls[(Num_layer - 1) * 6 + 3])) * self.spr_distance[Num_layer - 1]
        self.Vec_31[len(self.Vec_31) - 1].pos = self.walls[(Num_layer - 1) * 6 + 4] + self.walls[
            (Num_layer - 1) * 6 + 5]
        self.Vec_31[len(self.Vec_31) - 1].axis = vector.norm(
            self.walls[(Num_layer - 1) * 6 + 0] + self.walls[(Num_layer - 1) * 6 + 1] - (
                    self.walls[(Num_layer - 1) * 6 + 4] + self.walls[(Num_layer - 1) * 6 + 5])) * self.spr_distance[Num_layer - 1]
        self.top_cycle[len(self.Vec_12) - 1].pos = (self.Vec_12[len(self.Vec_12) - 1].pos + self.Vec_23[
            len(self.Vec_23) - 1].pos + self.Vec_31[len(self.Vec_31) - 1].pos) / 3
        self.top_cycle[len(self.Vec_12) - 1].axis = vector.cross(self.Vec_23[len(self.Vec_23) - 1].axis,
                                                                 self.Vec_12[len(self.Vec_12) - 1].axis)

    def update_upper_layers(self, Layer_num_start):
        Num_total = int(self.walls.__len__() / 6)
        Num_layer = Layer_num_start + 1
        # get totation matrix r = I + k + np.square(k) * ((1 - c) / (s ** 2))
        for i in range(Num_layer, Num_total):
            #print("start updatae layer:{0} out of {1}".format(i, Num_total))
            if self.spr_distance[i - 1] == self.spr_distance[i]:
                #print("self.spr_distance[i - 1] == self.spr_distance[i]")
                len_array = self.get_spring_len(i)
                org_bottom_vec = vector.cross(self.walls[(i) * 6 + 4] - self.walls[(i) * 6 + 0],
                                              self.walls[(i) * 6 + 2] - self.walls[(i) * 6 + 0])
                self.walls[(i) * 6 + 0] = self.walls[(i - 1) * 6 + 0] + self.walls[(i - 1) * 6 + 1]
                self.walls[(i) * 6 + 2] = self.walls[(i - 1) * 6 + 2] + self.walls[(i - 1) * 6 + 3]
                self.walls[(i) * 6 + 4] = self.walls[(i - 1) * 6 + 4] + self.walls[(i - 1) * 6 + 5]

                new_bottom_vec = vector.cross(self.walls[(i) * 6 + 4] - self.walls[(i) * 6 + 0],
                                              self.walls[(i) * 6 + 2] - self.walls[(i) * 6 + 0])

                rotation_matrix = get_rotation_matrix_from_vector(org_bottom_vec.hat, new_bottom_vec.hat)
                new_spr_array = np.matmul(rotation_matrix, self.walls[(i) * 6 + 1].hat.value)
                self.walls[(i) * 6 + 1] = vec(new_spr_array[0], new_spr_array[1], new_spr_array[2]) * len_array[0]
                self.walls[(i) * 6 + 3] = vec(new_spr_array[0], new_spr_array[1], new_spr_array[2]) * len_array[1]
                self.walls[(i) * 6 + 5] = vec(new_spr_array[0], new_spr_array[1], new_spr_array[2]) * len_array[2]
                if self.shift_angle[i] == 0:
                    self.update_current_layer_segment(i)

                else:
                    new_V1, new_V2, new_V3 = self.swift_base_points(i)
                    self.walls[(i) * 6 + 0] = new_V1
                    self.walls[(i) * 6 + 2] = new_V2
                    self.walls[(i) * 6 + 4] = new_V3
                    self.update_current_layer_segment(i)

            else:  # self.spr_distance[i - 1] != self.spr_distance[i]
                #print("self.spr_distance[i - 1] != self.spr_distance[i]")
                len_array = self.get_spring_len(i)
                pre_layer_top_center_point = self.get_one_layer_center_positions(i - 1)
                org_bottom_vec = vector.cross(self.walls[(i) * 6 + 4] - self.walls[(i) * 6 + 0],
                                              self.walls[(i) * 6 + 2] - self.walls[(i) * 6 + 0])

                pre_spring1_end = self.walls[(i - 1) * 6 + 0] + self.walls[(i - 1) * 6 + 1]
                pre_spring2_end = self.walls[(i - 1) * 6 + 2] + self.walls[(i - 1) * 6 + 3]
                pre_spring3_end = self.walls[(i - 1) * 6 + 4] + self.walls[(i - 1) * 6 + 5]
                #pre_layer_top_center_point = (pre_spring1_end + pre_spring2_end + pre_spring3_end)/3

                new_start_point1 = (pre_spring1_end - pre_layer_top_center_point).hat * self.spr_distance[i] / np.sqrt(3) + pre_layer_top_center_point
                new_start_point2 = (pre_spring2_end - pre_layer_top_center_point).hat * self.spr_distance[i] / np.sqrt(3) + pre_layer_top_center_point
                new_start_point3 = (pre_spring3_end - pre_layer_top_center_point).hat * self.spr_distance[i] / np.sqrt(3) + pre_layer_top_center_point
                self.walls[(i) * 6 + 0] = new_start_point1
                self.walls[(i) * 6 + 2] = new_start_point2
                self.walls[(i) * 6 + 4] = new_start_point3

                new_bottom_vec = vector.cross(self.walls[(i) * 6 + 4] - self.walls[(i) * 6 + 0],
                                              self.walls[(i) * 6 + 2] - self.walls[(i) * 6 + 0])
                rotation_matrix = get_rotation_matrix_from_vector(org_bottom_vec.hat, new_bottom_vec.hat)
                new_spr_array = np.matmul(rotation_matrix, self.walls[(i) * 6 + 1].hat.value)
                self.walls[(i) * 6 + 1] = vec(new_spr_array[0], new_spr_array[1], new_spr_array[2]) * len_array[0]
                self.walls[(i) * 6 + 3] = vec(new_spr_array[0], new_spr_array[1], new_spr_array[2]) * len_array[1]
                self.walls[(i) * 6 + 5] = vec(new_spr_array[0], new_spr_array[1], new_spr_array[2]) * len_array[2]

                if self.shift_angle[i] == 0:
                    self.update_current_layer_segment(i)

                else:
                    new_V1, new_V2, new_V3 = self.swift_base_points(i)
                    self.walls[(i) * 6 + 0] = new_V1
                    self.walls[(i) * 6 + 2] = new_V2
                    self.walls[(i) * 6 + 4] = new_V3
                    self.update_current_layer_segment(i)


    def update_current_layer_segment(self, Num_layer):
        self.spring1[Num_layer].axis = self.walls[(Num_layer) * 6 + 1]
        self.spring2[Num_layer].axis = self.walls[(Num_layer) * 6 + 3]
        self.spring3[Num_layer].axis = self.walls[(Num_layer) * 6 + 5]
        self.spring1[Num_layer].pos = self.walls[(Num_layer) * 6 + 0]
        self.spring2[Num_layer].pos = self.walls[(Num_layer) * 6 + 2]
        self.spring3[Num_layer].pos = self.walls[(Num_layer) * 6 + 4]
        # self.springC[Num_layer].axis = (self.walls[(Num_layer) * 6 + 1] + self.walls[
        # (Num_layer) * 6 + 3] + self.walls[(Num_layer) * 6 + 5]) / 3

        # self.springC[Num_layer].pos = (self.walls[(Num_layer ) * 6 + 0] + self.walls[
        # (Num_layer) * 6 + 2] + self.walls[(Num_layer ) * 6 + 4]) / 3

        self.Vec_12[Num_layer].pos = self.walls[(Num_layer) * 6 + 0] + self.walls[(Num_layer) * 6 + 1]
        self.Vec_12[Num_layer].axis = vector.norm(self.walls[(Num_layer) * 6 + 2] + self.walls[(Num_layer) * 6 + 3] - (
                self.walls[(Num_layer) * 6 + 0] + self.walls[(Num_layer) * 6 + 1])) * self.spr_distance[Num_layer]
        self.Vec_23[Num_layer].pos = self.walls[(Num_layer) * 6 + 2] + self.walls[(Num_layer) * 6 + 3]
        self.Vec_23[Num_layer].axis = vector.norm(self.walls[(Num_layer) * 6 + 4] + self.walls[(Num_layer) * 6 + 5] - (
                self.walls[(Num_layer) * 6 + 2] + self.walls[(Num_layer) * 6 + 3])) * self.spr_distance[Num_layer]
        self.Vec_31[Num_layer].pos = self.walls[(Num_layer) * 6 + 4] + self.walls[(Num_layer) * 6 + 5]
        self.Vec_31[Num_layer].axis = vector.norm(self.walls[(Num_layer) * 6 + 0] + self.walls[(Num_layer) * 6 + 1] - (
                self.walls[(Num_layer) * 6 + 4] + self.walls[(Num_layer) * 6 + 5])) * self.spr_distance[Num_layer]

        self.top_cycle[Num_layer].pos = (self.Vec_12[Num_layer].pos + self.Vec_23[Num_layer].pos + self.Vec_31[
            Num_layer].pos) / 3
        self.top_cycle[Num_layer].axis = vector.cross(self.Vec_23[Num_layer].axis, self.Vec_12[Num_layer].axis)

    def add_one_layer_on_top(self):
        Num_layer = int(self.walls.__len__() / 6)
        spring1_pos = self.walls[(Num_layer - 1) * 6 + 0]
        spring1_axis = self.walls[(Num_layer - 1) * 6 + 1]
        spring1_end = spring1_pos + spring1_axis
        spring2_pos = self.walls[(Num_layer - 1) * 6 + 2]
        spring2_axis = self.walls[(Num_layer - 1) * 6 + 3]
        spring2_end = spring2_pos + spring2_axis
        spring3_pos = self.walls[(Num_layer - 1) * 6 + 4]
        spring3_axis = self.walls[(Num_layer - 1) * 6 + 5]
        spring3_end = spring3_pos + spring3_axis

        initial_vec = vector.cross((spring3_end - spring1_end), (spring2_end - spring1_end)).hat * self.ini_len    # Problem
        Num_layer_new = int(self.walls.__len__() / 6)+1

        if self.spr_distance[Num_layer] == self.spr_distance[Num_layer - 1]:
            if self.shift_angle[Num_layer] == 0:
                self.walls.append(spring1_end)
                self.walls.append(initial_vec)
                self.walls.append(spring2_end)
                self.walls.append(initial_vec)
                self.walls.append(spring3_end)
                self.walls.append(initial_vec)
                self.spring_len.append([self.ini_len, self.ini_len, self.ini_len])
            else:
                Rotation_angle = self.shift_angle[Num_layer]
                N1, N2, N3 = swift_three_points( spring1_end, spring2_end, spring1_end,Rotation_angle, initial_vec.hat)
                self.walls.append(N1)
                self.walls.append(initial_vec)
                self.walls.append(N2)
                self.walls.append(initial_vec)
                self.walls.append(N3)
                self.walls.append(initial_vec)
                self.spring_len.append([self.ini_len, self.ini_len, self.ini_len])

        else:

                top_center_point = self.get_top_center_positions()
                new_start_point1 = (spring1_end - top_center_point).hat * self.spr_distance[Num_layer] / np.sqrt(3) + top_center_point
                new_start_point2 = (spring2_end - top_center_point).hat * self.spr_distance[Num_layer] / np.sqrt(3) + top_center_point
                new_start_point3 = (spring3_end - top_center_point).hat * self.spr_distance[Num_layer] / np.sqrt(3) + top_center_point
                if self.shift_angle[Num_layer] == 0:
                    self.walls.append(new_start_point1)
                    self.walls.append(initial_vec)
                    self.walls.append(new_start_point2)
                    self.walls.append(initial_vec)
                    self.walls.append(new_start_point3)
                    self.walls.append(initial_vec)
                    self.spring_len.append([self.ini_len, self.ini_len, self.ini_len])
                else:
                    Rotation_angle = self.shift_angle[Num_layer]
                    N1, N2, N3 = swift_three_points(new_start_point1, new_start_point2, new_start_point3, Rotation_angle,
                                                    initial_vec.hat)
                    self.walls.append(N1)
                    self.walls.append(initial_vec)
                    self.walls.append(N2)
                    self.walls.append(initial_vec)
                    self.walls.append(N3)
                    self.walls.append(initial_vec)
                    self.spring_len.append([self.ini_len, self.ini_len, self.ini_len])

        self.spring1[len(self.spring1):] = [cylinder(pos=self.walls[(Num_layer_new - 1) * 6 + 0],  # (x1, y1, 0),
                                                  axis=self.walls[(Num_layer_new - 1) * 6 + 1],
                                                  # vector(0, 0, self.spr_len),
                                                  #thickness=self.spr_len_thickness, coils=self.spr_coils,
                                                  radius=self.spr_len_thickness, color=self.spr_color_vec,
                                                  opacity=self.lateral_opacity)]

        self.spring2[len(self.spring2):] = [cylinder(pos=self.walls[(Num_layer_new - 1) * 6 + 2],  # vector(x2, y2, 0),
                                                  axis=self.walls[(Num_layer_new - 1) * 6 + 3],
                                                  # vector(0, 0, self.spr_len),
                                                  #thickness=self.spr_len_thickness, coils=self.spr_coils,
                                                  radius=self.spr_len_thickness, color=self.spr_color_vec,
                                                  opacity=self.lateral_opacity)]

        self.spring3[len(self.spring3):] = [cylinder(pos=self.walls[(Num_layer_new - 1) * 6 + 4],  # vector(x3, y3, 0),
                                                  axis=self.walls[(Num_layer_new - 1) * 6 + 5],
                                                  # vector(0, 0, self.spr_len),
                                                  #thickness=self.spr_len_thickness, coils=self.spr_coils,
                                                  radius=self.spr_len_thickness, color=self.spr_color_vec,
                                                  opacity=self.lateral_opacity)]

        # self.springC[len(self.springC):] = [helix(pos = ((self.walls[(Num_layer_new - 1)*6 + 0] +
        #                                                   self.walls[(Num_layer_new - 1)*6 + 2] +
        #                                                   self.walls[(Num_layer_new - 1)*6 + 4] )/3),
        #                                         axis = ((self.walls[(Num_layer_new - 1)*6 + 1] +
        #                                                  self.walls[(Num_layer_new - 1)*6 + 3] +
        #                                                  self.walls[(Num_layer_new - 1)*6 + 5] )/3),
        #                                         thickness = self.spr_len_thickness, coils = self.spr_coils,
        #                                         radius = self.spr_radius, color = vec(19/255, 153/255, 36/255) )]

        self.Vec_12[len(self.Vec_12):] = [
            cylinder(pos=self.walls[(Num_layer_new - 1) * 6 + 0] + self.walls[(Num_layer_new - 1) * 6 + 1],
                     axis=vector.norm(
                         self.walls[(Num_layer_new - 1) * 6 + 2] + self.walls[(Num_layer_new - 1) * 6 + 3] - (
                                 self.walls[(Num_layer_new - 1) * 6 + 0] + self.walls[
                             (Num_layer_new - 1) * 6 + 1])) * self.spr_distance[Num_layer_new - 1],
                     thickness=self.unit_len_thickness,
                     radius=self.unit_len_thickness, color=self.spr_color_vec, opacity=0)]

        self.Vec_23[len(self.Vec_23):] = [
            cylinder(pos=self.walls[(Num_layer_new - 1) * 6 + 2] + self.walls[(Num_layer_new - 1) * 6 + 3],
                     axis=vector.norm(
                         self.walls[(Num_layer_new - 1) * 6 + 4] + self.walls[(Num_layer_new - 1) * 6 + 5] - (
                                 self.walls[(Num_layer_new - 1) * 6 + 2] + self.walls[
                             (Num_layer_new - 1) * 6 + 3])) * self.spr_distance[Num_layer_new - 1],
                     thickness=self.unit_len_thickness,
                     radius=self.unit_len_thickness, color=self.spr_color_vec, opacity=0)]

        self.Vec_31[len(self.Vec_31):] = [
            cylinder(pos=self.walls[(Num_layer_new - 1) * 6 + 4] + self.walls[(Num_layer_new - 1) * 6 + 5],
                     axis=vector.norm(
                         self.walls[(Num_layer_new - 1) * 6 + 0] + self.walls[(Num_layer_new - 1) * 6 + 1] - (
                                 self.walls[(Num_layer_new - 1) * 6 + 4] + self.walls[
                             (Num_layer_new - 1) * 6 + 5])) * self.spr_distance[Num_layer_new - 1],
                     thickness=self.unit_len_thickness,
                     radius=self.unit_len_thickness, color=self.spr_color_vec, opacity=0)]

        ring_rad = self.spr_distance[Num_layer_new - 1] / np.sqrt(3)
        self.top_cycle[len(self.top_cycle):] = [ring(pos=(self.Vec_12[len(self.Vec_12) - 1].pos + self.Vec_23[
            len(self.Vec_23) - 1].pos + self.Vec_31[len(self.Vec_31) - 1].pos) / 3,
                                                     axis=vector.cross(self.Vec_23[len(self.Vec_23) - 1].axis,
                                                                       self.Vec_12[len(self.Vec_12) - 1].axis),
                                                     radius=ring_rad, thickness=self.unit_len_thickness,
                                                     color=self.spr_color_vec, opacity=self.lateral_opacity )] #self.lateral_opacity)]
        if (len(self.Vec_31) - 1) % self.module_num == 0:
            self.top_cycle[len(self.top_cycle) - 1].color = vec(0 / 255, 102 / 255, 0 / 255)
            # self.update_top_segment()


    def add_one_module_on_top(self, initial_len):
        dt = initial_len / self.module_num
        for i in range(self.module_num):
            self.add_one_layer_on_top()
            self.increase_all_sides(dt)

    def increase_one_side(self, stepLength, SprNum):
        # only increase the top layer
        print("start to increase spring{0}".format(SprNum))
        Num_layer = int(self.walls.__len__() / 6)
        spring1_pos = self.walls[(Num_layer - 1) * 6 + 0]
        spring1_axis = self.walls[(Num_layer - 1) * 6 + 1]
        spring1_len = spring1_axis.mag
        spring1_end = spring1_pos + spring1_axis
        spring2_pos = self.walls[(Num_layer - 1) * 6 + 2]
        spring2_axis = self.walls[(Num_layer - 1) * 6 + 3]
        spring2_len = spring2_axis.mag
        spring2_end = spring2_pos + spring2_axis
        spring3_pos = self.walls[(Num_layer - 1) * 6 + 4]
        spring3_axis = self.walls[(Num_layer - 1) * 6 + 5]
        spring3_len = spring3_axis.mag
        spring3_end = spring3_pos + spring3_axis
        spring1_len = self.spring_len[Num_layer - 1][0]
        spring2_len = self.spring_len[Num_layer - 1][1]
        spring3_len = self.spring_len[Num_layer - 1][2]
        '''
        vec1_to_2 = spring2_end - spring1_end
        vec2_to_3 = spring3_end - spring2_end
        vec3_to_1 = spring1_end - spring3_end
        center_top = (spring1_end + spring2_end + spring3_end)/3
        vec_bottom_unit = vector.cross((spring3_end - spring1_end), (spring2_end - spring1_end)).norm()
        vec_top_unit    = vector.cross((spring3_end - spring1_end), (spring2_end - spring1_end)).norm()
        '''
        # rotation_angle = math.asin(stepLength/2/self.spr_distance)*2
        if (SprNum == 1):
            spring1_len = spring1_len + stepLength
            # spring1_end_tran = spring1_end - spring2_end
            # rotate(spring1_end_tran, angle = rotation_angle, axis = vec2_to_3 )
        elif (SprNum == 2):
            spring2_len = spring2_len + stepLength
        else:  # SprNum == 3
            spring3_len = spring3_len + stepLength

        max_len = max(spring1_len, spring2_len, spring3_len)
        min_len = min(spring1_len, spring2_len, spring3_len)
        dif_len = max(spring1_len - min_len, spring2_len - min_len, spring3_len - min_len)
        if self.max_len < max_len and self.max_dif_len < dif_len:
            print("spring length out of limitation in Func: increase_one_side")
            return 0  # indicate no grow

        switch_num = switch_for_cal_vec(spring1_len, spring2_len, spring3_len)
        # print("switch_num = {}".format(switch_num))
        spring_len_array = [spring1_len, spring2_len, spring3_len]
        spring_pos_array = [spring1_pos, spring2_pos, spring3_pos]
        spring_len_after_switch = np.roll(spring_len_array, switch_num)
        spring_pos_after_switch = np.roll(spring_pos_array, switch_num)

        list_spring_unswitch_axix = calculation_spring_vector(spring_pos_after_switch[0], spring_pos_after_switch[1],
                                                              spring_pos_after_switch[2], spring_len_after_switch[0],
                                                              spring_len_after_switch[1], spring_len_after_switch[2],
                                                              self.spr_distance[Num_layer])

        list_spring_axix = np.roll(list_spring_unswitch_axix, switch_num * -1)

        self.walls[(Num_layer - 1) * 6 + 1] = list_spring_axix[0]
        self.walls[(Num_layer - 1) * 6 + 3] = list_spring_axix[1]
        self.walls[(Num_layer - 1) * 6 + 5] = list_spring_axix[2]
        self.update_top_segment()
        if (SprNum == 1):
            self.spring_len[Num_layer - 1][0] = spring1_len + stepLength
        elif (SprNum == 2):
            self.spring_len[Num_layer - 1][1] = spring2_len + stepLength
        else:  # SprNum == 3
            self.spring_len[Num_layer - 1][2] = spring3_len + stepLength
        return 1

    def increase_all_sides(self, stepLength):
        # increase only top layer
        Num_layer = int(self.walls.__len__() / 6)
        spring1_pos = self.walls[(Num_layer - 1) * 6 + 0]
        spring1_axis = self.walls[(Num_layer - 1) * 6 + 1]
        spring1_len = spring1_axis.mag
        spring1_end = spring1_pos + spring1_axis
        spring2_pos = self.walls[(Num_layer - 1) * 6 + 2]
        spring2_axis = self.walls[(Num_layer - 1) * 6 + 3]
        spring2_len = spring2_axis.mag
        spring2_end = spring2_pos + spring2_axis
        spring3_pos = self.walls[(Num_layer - 1) * 6 + 4]
        spring3_axis = self.walls[(Num_layer - 1) * 6 + 5]
        spring3_len = spring3_axis.mag
        spring3_end = spring3_pos + spring3_axis
        [spring1_len, spring2_len, spring3_len] = self.get_spring_len(Num_layer - 1)
        vec = spring1_axis.hat
        max_len = max(spring1_len + stepLength, spring2_len + stepLength, spring3_len + stepLength)
        if self.max_len < max_len:
            return 0  # indicate no grow

        spring1_axis_new = vec * (spring1_len + stepLength)
        spring2_axis_new = vec * (spring2_len + stepLength)
        spring3_axis_new = vec * (spring3_len + stepLength)
        self.walls[(Num_layer - 1) * 6 + 1] = spring1_axis_new
        self.walls[(Num_layer - 1) * 6 + 3] = spring2_axis_new
        self.walls[(Num_layer - 1) * 6 + 5] = spring3_axis_new
        self.spring_len[(Num_layer - 1)] = [spring1_len + stepLength, spring2_len + stepLength,
                                            spring3_len + stepLength]
        self.update_top_segment()
        return 1

    def increase_mid_layer_one_side(self, stepLength, SprNumT):
        # SprNumT is from 0
        LayerNum = int((SprNumT) / 3)
        SprNum = SprNumT % 3
        spring1_len = self.walls[(LayerNum) * 6 + 1].mag
        spring2_len = self.walls[(LayerNum) * 6 + 3].mag
        spring3_len = self.walls[(LayerNum) * 6 + 5].mag
        [spring1_len, spring2_len, spring3_len] = self.get_spring_len(LayerNum)

        k = [0, 0, 0]
        k[SprNum] = 1
        spr1_len = spring1_len + k[0] * stepLength
        spr2_len = spring2_len + k[1] * stepLength
        spr3_len = spring3_len + k[2] * stepLength
        return self.change_inner_layer_len(LayerNum, spr1_len, spr2_len, spr3_len)

    def increasing_one_side_by_module(self, stepLength, SprNum, Module_Num):
        # SprNum from 0-
        if SprNum > 2 or SprNum < 0:
            print("ERROR: SprNum value range should be [0~2]")
            return 0
        dt = stepLength / self.module_num
        #total_layerN = int(self.walls.__len__() / 6)
        #total_moduleN = total_layerN/self.module_num
        start_layerN = Module_Num * self.module_num * 3
        print("start_layerN = {}, SprNum = {}, self.module_num ={}".format(start_layerN, SprNum, self.module_num))
        for i in range(start_layerN + SprNum +1, start_layerN + self.module_num * 3 + SprNum, 3):
            res = self.increase_mid_layer_one_side(dt, i)
            print("spring_num = {}".format(i))
            if res == 0:
                print("not increase any more, out of limitation")
            # sleep(1)

    def change_inner_layer_len(self, LayerNum, spr1_len, spr2_len, spr3_len):
        # layerNum from 0 to Num_layer = int(self.walls.__len__()/6) -1
        ToNum_layer = int(self.walls.__len__() / 6)
        if LayerNum < 0:
            return 0

        Num_layer = LayerNum
        spring1_pos = self.walls[(Num_layer) * 6 + 0]
        spring1_axis = self.walls[(Num_layer) * 6 + 1]
        spring1_len = spring1_axis.mag
        spring1_end = spring1_pos + spring1_axis
        spring2_pos = self.walls[(Num_layer) * 6 + 2]
        spring2_axis = self.walls[(Num_layer) * 6 + 3]
        spring2_len = spring2_axis.mag
        spring2_end = spring2_pos + spring2_axis
        spring3_pos = self.walls[(Num_layer) * 6 + 4]
        spring3_axis = self.walls[(Num_layer) * 6 + 5]
        spring3_len = spring3_axis.mag
        spring3_end = spring3_pos + spring3_axis

        ve3 = spring3_pos - spring1_pos
        ve2 = spring2_pos - spring1_pos
        bottom_unit_vec = vector.cross(ve3, ve2).hat
        top_unit_vec = spring3_axis.hat
        # after_rotation_vec = rotate(spring1_end_tran, angle=rotation_angle, axis=vec2_to_3)

        list_axis = self.calculation_spring_axis(spr1_len, spr2_len, spr3_len, spring1_pos, spring2_pos, spring3_pos, Num_layer)
        if list_axis != 0:
            self.walls[(Num_layer) * 6 + 1] = list_axis[0]
            self.walls[(Num_layer) * 6 + 3] = list_axis[1]
            self.walls[(Num_layer) * 6 + 5] = list_axis[2]
            self.spring_len[Num_layer] = [spr1_len, spr2_len, spr3_len]
            self.update_current_layer_segment(Num_layer)
            self.update_upper_layers(Num_layer)
            return 1
        else:
            print("the spring len increase outside limitation")
            return 0

    def change_inner_module_lens(self, Module_Num, len1, len2, len3):
        len1_spr = len1 / self.module_num
        len2_spr = len2 / self.module_num
        len3_spr = len3 / self.module_num
        for i in range(Module_Num * self.module_num+1, (Module_Num + 1) * self.module_num+1):
            # print("inrease layer Num: {}, module Num: {}".format(i,Module_Num))
            self.change_inner_layer_len(i, len1_spr, len2_spr, len3_spr)

    def swift_base_points(self, Num_layer):
        #function just swift the base three points of the one layer
        i = Num_layer
        rot_axis = self.walls[(i) * 6 + 1].hat  # unit
        rotation_angle = self.shift_angle[i]
        pre_layer_top_center_point = self.get_one_layer_center_positions(i - 1)
        V1 = self.walls[(i) * 6 + 0] - pre_layer_top_center_point
        V2 = self.walls[(i) * 6 + 2] - pre_layer_top_center_point
        V3 = self.walls[(i) * 6 + 4] - pre_layer_top_center_point
        new_V1 = rotate(V1, angle=rotation_angle, axis=rot_axis) + pre_layer_top_center_point
        new_V2 = rotate(V2, angle=rotation_angle, axis=rot_axis) + pre_layer_top_center_point
        new_V3 = rotate(V3, angle=rotation_angle, axis=rot_axis) + pre_layer_top_center_point
        return [new_V1, new_V2, new_V3]



    def get_top_ending_position(self):
        Num_layer = int(self.walls.__len__() / 6)
        spring1_pos = self.walls[(Num_layer - 1) * 6 + 0]
        spring1_axis = self.walls[(Num_layer - 1) * 6 + 1]
        spring1_len = spring1_axis.mag
        spring1_end = spring1_pos + spring1_axis
        spring2_pos = self.walls[(Num_layer - 1) * 6 + 2]
        spring2_axis = self.walls[(Num_layer - 1) * 6 + 3]
        spring2_len = spring2_axis.mag
        spring2_end = spring2_pos + spring2_axis
        spring3_pos = self.walls[(Num_layer - 1) * 6 + 4]
        spring3_axis = self.walls[(Num_layer - 1) * 6 + 5]
        spring3_len = spring3_axis.mag
        spring3_end = spring3_pos + spring3_axis
        return [spring1_end, spring2_end, spring3_end]

    def get_lowest_point_Num(self):
        [d1, d2, d3] = self.get_top_ending_position()
        dz = []
        dz.append(d1._z)
        dz.append(d2._z)
        dz.append(d3._z)
        dz = np.array(dz)
        return np.where(dz == np.min(dz))[0][0]

    def get_highest_point_Num(self):
        [d1, d2, d3] = self.get_top_ending_position()
        dz = []
        dz.append(d1._z)
        dz.append(d2._z)
        dz.append(d3._z)
        dz = np.array(dz)
        return np.where(dz == np.max(dz))[0][0]

    def get_top_direction(self):
        Num_layer = int(self.walls.__len__() / 6)
        spring1_pos = self.walls[(Num_layer - 1) * 6 + 0]
        spring1_axis = self.walls[(Num_layer - 1) * 6 + 1]
        spring1_len = spring1_axis.mag
        spring1_end = spring1_pos + spring1_axis
        spring2_pos = self.walls[(Num_layer - 1) * 6 + 2]
        spring2_axis = self.walls[(Num_layer - 1) * 6 + 3]
        spring2_len = spring2_axis.mag
        spring2_end = spring2_pos + spring2_axis
        spring3_pos = self.walls[(Num_layer - 1) * 6 + 4]
        spring3_axis = self.walls[(Num_layer - 1) * 6 + 5]
        spring3_len = spring3_axis.mag
        spring3_end = spring3_pos + spring3_axis
        ve3 = spring3_end - spring1_end
        ve2 = spring2_end - spring1_end
        bottom_vec = vector.cross(ve3, ve2).hat
        return bottom_vec

    def get_module_end_direction(self, Module_Num):
        Num_layer = Module_Num * self.module_num
        spring1_pos = self.walls[(Num_layer - 1) * 6 + 0]
        spring1_axis = self.walls[(Num_layer - 1) * 6 + 1]
        spring1_len = spring1_axis.mag
        spring1_end = spring1_pos + spring1_axis
        spring2_pos = self.walls[(Num_layer - 1) * 6 + 2]
        spring2_axis = self.walls[(Num_layer - 1) * 6 + 3]
        spring2_len = spring2_axis.mag
        spring2_end = spring2_pos + spring2_axis
        spring3_pos = self.walls[(Num_layer - 1) * 6 + 4]
        spring3_axis = self.walls[(Num_layer - 1) * 6 + 5]
        spring3_len = spring3_axis.mag
        spring3_end = spring3_pos + spring3_axis
        ve3 = spring3_end - spring1_end
        ve2 = spring2_end - spring1_end
        bottom_vec = vector.cross(ve3, ve2).hat
        return bottom_vec

    def get_module_impact_from_vector(self, Module_Num, input_vector):
        end_vector = self.get_module_end_direction(Module_Num)
        vec_factor = vector.dot(end_vector.hat,input_vector.hat)
        return vec_factor

    def get_spring_len(self, layer_num):
        Num_layer = layer_num
        spring1_len = self.walls[(Num_layer) * 6 + 1].mag
        spring2_len = self.walls[(Num_layer) * 6 + 3].mag
        spring3_len = self.walls[(Num_layer) * 6 + 5].mag

        # print("lens are {0} {1} {2} ".format(spring1_len, spring2_len, spring3_len))
        # return np.array([spring1_len, spring2_len,spring3_len])
        return np.array(self.spring_len[Num_layer])

    def get_center_positions(self, Num_layer):
        # Num_layer = int(self.walls.__len__()/6)
        spring1_pos = self.walls[(Num_layer) * 6 + 0]
        spring1_axis = self.walls[(Num_layer) * 6 + 1]
        spring1_end = spring1_pos + spring1_axis
        spring2_pos = self.walls[(Num_layer) * 6 + 2]
        spring2_axis = self.walls[(Num_layer) * 6 + 3]
        spring2_end = spring2_pos + spring2_axis
        spring3_pos = self.walls[(Num_layer) * 6 + 4]
        spring3_axis = self.walls[(Num_layer) * 6 + 5]
        spring3_end = spring3_pos + spring3_axis
        top_center_up = (spring1_end + spring2_end + spring3_end) / 3
        top_center_down = (spring1_pos + spring2_pos + spring3_pos) / 3
        return [top_center_up, top_center_down]

    def get_top_center_positions(self):
        Num_layer = int(self.walls.__len__() / 6) - 1
        spring1_pos = self.walls[(Num_layer) * 6 + 0]
        spring1_axis = self.walls[(Num_layer) * 6 + 1]
        spring1_end = spring1_pos + spring1_axis
        spring2_pos = self.walls[(Num_layer) * 6 + 2]
        spring2_axis = self.walls[(Num_layer) * 6 + 3]
        spring2_end = spring2_pos + spring2_axis
        spring3_pos = self.walls[(Num_layer) * 6 + 4]
        spring3_axis = self.walls[(Num_layer) * 6 + 5]
        spring3_end = spring3_pos + spring3_axis
        top_center_up = (spring1_end + spring2_end + spring3_end) / 3
        top_center_down = (spring1_pos + spring2_pos + spring3_pos) / 3
        return top_center_up

    def get_one_layer_center_positions(self, Num_layer):

        spring1_pos = self.walls[(Num_layer) * 6 + 0]
        spring1_axis = self.walls[(Num_layer) * 6 + 1]
        spring1_end = spring1_pos + spring1_axis
        spring2_pos = self.walls[(Num_layer) * 6 + 2]
        spring2_axis = self.walls[(Num_layer) * 6 + 3]
        spring2_end = spring2_pos + spring2_axis
        spring3_pos = self.walls[(Num_layer) * 6 + 4]
        spring3_axis = self.walls[(Num_layer) * 6 + 5]
        spring3_end = spring3_pos + spring3_axis
        top_center_up = (spring1_end + spring2_end + spring3_end) / 3
        top_center_down = (spring1_pos + spring2_pos + spring3_pos) / 3
        return top_center_up


    def Distance_from_center_to_rod_by_layer(self, LayerNum, rod_pos, rod_end, rod_radius):

        top_position = self.get_center_positions(LayerNum)
        return calulation_line_to_point(top_position[0], rod_pos, rod_end) - rod_radius

    def Distance_from_top_to_target(self, *args, **kwargs):
        if len(kwargs) == 2:
            target = 'ball'
            _ball_pos = kwargs['ball_pos']
            _ball_radius = kwargs['ball_rad']
            return self.Distance_from_top_to_round_shape(_ball_pos, _ball_radius)
        elif len(kwargs) == 3:
            target = 'rod'
            _rod_pos = kwargs['rod_pos']
            _rod_axis = kwargs['rod_axis']
            _rod_end = _rod_pos + _rod_axis
            _rod_radius = kwargs['rod_rad']
            return self.Distance_from_top_to_rod_shape(_rod_pos, _rod_end, _rod_radius)

    def Distance_from_mid_layer_to_target(self, LayerNum, *args, **kwargs):
        if len(kwargs) == 2:
            target = 'ball'
            _ball_pos = kwargs['ball_pos']
            _ball_radius = kwargs['ball_rad']
            return self.Distance_from_mid_layer_to_round_shape(LayerNum, _ball_pos, _ball_radius)
        elif len(kwargs) == 3:
            target = 'rod'
            _rod_pos = kwargs['rod_pos']
            _rod_axis = kwargs['rod_axis']
            _rod_end = _rod_pos + _rod_axis
            _rod_radius = kwargs['rod_rad']
            return self.Distance_from_mid_layer_to_rod_shape(LayerNum, _rod_pos, _rod_end, _rod_radius)

    def Distance_from_top_to_round_shape(self, point_pos, point_radius):
        Num_layer = int(self.walls.__len__() / 6)
        spring1_pos = self.walls[(Num_layer - 1) * 6 + 0]
        spring1_axis = self.walls[(Num_layer - 1) * 6 + 1]
        spring1_end = spring1_pos + spring1_axis
        spring2_pos = self.walls[(Num_layer - 1) * 6 + 2]
        spring2_axis = self.walls[(Num_layer - 1) * 6 + 3]
        spring2_end = spring2_pos + spring2_axis
        spring3_pos = self.walls[(Num_layer - 1) * 6 + 4]
        spring3_axis = self.walls[(Num_layer - 1) * 6 + 5]
        spring3_end = spring3_pos + spring3_axis
        dis_spr1 = (spring1_end - point_pos).mag - point_radius
        dis_spr2 = (spring2_end - point_pos).mag - point_radius
        dis_spr3 = (spring3_end - point_pos).mag - point_radius
        dis_list = []
        dis_list.append(dis_spr1)
        dis_list.append(dis_spr2)
        dis_list.append(dis_spr3)
        return dis_list

    def Distance_from_mid_layer_to_round_shape(self, LayerNum, point_pos, point_radius):
        Num_layer = LayerNum
        spring1_pos = self.walls[(LayerNum) * 6 + 0]
        spring1_axis = self.walls[(LayerNum) * 6 + 1]
        spring1_end = spring1_pos + spring1_axis
        spring2_pos = self.walls[(LayerNum) * 6 + 2]
        spring2_axis = self.walls[(LayerNum) * 6 + 3]
        spring2_end = spring2_pos + spring2_axis
        spring3_pos = self.walls[(LayerNum) * 6 + 4]
        spring3_axis = self.walls[(LayerNum) * 6 + 5]
        spring3_end = spring3_pos + spring3_axis
        dis_spr1 = (spring1_end - point_pos).mag - point_radius
        dis_spr2 = (spring2_end - point_pos).mag - point_radius
        dis_spr3 = (spring3_end - point_pos).mag - point_radius
        dis_list = []
        dis_list.append(dis_spr1)
        dis_list.append(dis_spr2)
        dis_list.append(dis_spr3)
        return dis_list

    def Distance_from_top_to_rod_shape(self, rod_ending1, rod_ending2, rod_radius):
        Num_layer = int(self.walls.__len__() / 6)
        spring1_pos = self.walls[(Num_layer - 1) * 6 + 0]
        spring1_axis = self.walls[(Num_layer - 1) * 6 + 1]
        spring1_end = spring1_pos + spring1_axis
        spring2_pos = self.walls[(Num_layer - 1) * 6 + 2]
        spring2_axis = self.walls[(Num_layer - 1) * 6 + 3]
        spring2_end = spring2_pos + spring2_axis
        spring3_pos = self.walls[(Num_layer - 1) * 6 + 4]
        spring3_axis = self.walls[(Num_layer - 1) * 6 + 5]
        spring3_end = spring3_pos + spring3_axis

        list = []
        dis_spring1 = calulation_line_to_point(spring1_end, rod_ending1, rod_ending2) - rod_radius
        dis_spring2 = calulation_line_to_point(spring2_end, rod_ending1, rod_ending2) - rod_radius
        dis_spring3 = calulation_line_to_point(spring3_end, rod_ending1, rod_ending2) - rod_radius
        list.append(dis_spring1)
        list.append(dis_spring2)
        list.append(dis_spring3)
        return list

    def Distance_from_mid_layer_to_rod_shape(self, LayerNum, rod_ending1, rod_ending2, rod_radius):
        Num_layer = LayerNum
        spring1_pos = self.walls[(LayerNum) * 6 + 0]
        spring1_axis = self.walls[(LayerNum) * 6 + 1]
        spring1_end = spring1_pos + spring1_axis
        spring2_pos = self.walls[(LayerNum) * 6 + 2]
        spring2_axis = self.walls[(LayerNum) * 6 + 3]
        spring2_end = spring2_pos + spring2_axis
        spring3_pos = self.walls[(LayerNum) * 6 + 4]
        spring3_axis = self.walls[(LayerNum) * 6 + 5]
        spring3_end = spring3_pos + spring3_axis

        list = []
        dis_spring1 = calulation_line_to_point(spring1_end, rod_ending1, rod_ending2) - rod_radius
        dis_spring2 = calulation_line_to_point(spring2_end, rod_ending1, rod_ending2) - rod_radius
        dis_spring3 = calulation_line_to_point(spring3_end, rod_ending1, rod_ending2) - rod_radius
        list.append(dis_spring1)
        list.append(dis_spring2)
        list.append(dis_spring3)
        return list

    def append_dist2target(self, dist_list):
        self.dist_springs2target.append([dist_list[0], dist_list[1], dist_list[2]])

    def calculation_spring_axis(self, spring1_len, spring2_len, spring3_len, spring1_pos, spring2_pos, spring3_pos, Num_layer):

        max_len = max(spring1_len, spring2_len, spring3_len)
        min_len = min(spring1_len, spring2_len, spring3_len)
        dif_len = max(spring1_len - min_len, spring2_len - min_len, spring3_len - min_len)
        if self.max_len < max_len and self.max_dif_len < dif_len:
            return 0  # indicate len out of limit

        switch_num = switch_for_cal_vec(spring1_len, spring2_len, spring3_len)
        # print("switch_num = {}".format(switch_num))
        spring_len_array = [spring1_len, spring2_len, spring3_len]
        spring_pos_array = [spring1_pos, spring2_pos, spring3_pos]
        spring_len_after_switch = np.roll(spring_len_array, switch_num)
        spring_pos_after_switch = np.roll(spring_pos_array, switch_num)

        list_spring_unswitch_axix = calculation_spring_vector(spring_pos_after_switch[0], spring_pos_after_switch[1],
                                                              spring_pos_after_switch[2], spring_len_after_switch[0],
                                                              spring_len_after_switch[1], spring_len_after_switch[2],
                                                              self.spr_distance[Num_layer])

        list_spring_axix = np.roll(list_spring_unswitch_axix, switch_num * -1)

        return [list_spring_axix[0], list_spring_axix[1], list_spring_axix[2]]

    def find_close_layer_point(self, *args, **kwargs):
        dist_list_all = []
        totall_layerN = int(self.walls.__len__() / 6)
        for i in range(totall_layerN):
            dist_list_all.append(self.Distance_from_mid_layer_to_target(LayerNum=i, *args, **kwargs))
        dist_list_all = np.array(dist_list_all)
        dist_list_all = dist_list_all.reshape(-1)
        print("dist_list_all:{}".format(dist_list_all))
        minN = np.where(dist_list_all == min(dist_list_all))
        close_dist = dist_list_all[minN]
        layer_Num = int(minN[0][0] / 3)
        print("layer_Num:{}".format(layer_Num))
        layer_dist = (self.Distance_from_mid_layer_to_target(LayerNum=layer_Num, *args, **kwargs))
        layer_dist = np.array(layer_dist)
        maxN = np.where(layer_dist == max(layer_dist))[0][0]  # 0-2
        print("maxN:{}".format(maxN))
        return layer_Num * 3 + maxN, close_dist

    def touchSense_close_layer_point_by_module(self, rodPos, rodEnd, rodRadius):
        dist_list_all = []
        layIndex = []
        totall_moduleN = int(self.walls.__len__() / 6 / self.module_num)
        totall_layerN = int(self.walls.__len__() / 6)
        for i in range(1, totall_layerN, 1):
            # print("module layer={}".format(i))
            dist_list_all.append(self.Distance_from_center_to_rod_by_layer(i, rodPos, rodEnd, rodRadius))
            layIndex.append(i)
        dist_list_all = np.array(dist_list_all)
        dist_list_all = dist_list_all.reshape(-1)
        # print("dist_list_all:{}".format(dist_list_all))
        minN = np.where(dist_list_all == min(dist_list_all))
        close_dist = dist_list_all[minN]
        layer_Num = layIndex[int(minN[0][0])]
        print("closest layer_Num:{}".format(layer_Num))
        dis_spring = self.Distance_from_mid_layer_to_rod_shape(layer_Num, rodPos, rodEnd, rodRadius)

        return [layer_Num, dis_spring]

    def touchSense_close_layer_point_by_module_from_new(self, module_number, rodPos, rodEnd, rodRadius):
        # start sensing from module_number
        dist_list_all = []
        layIndex = []
        totall_moduleN = int(self.walls.__len__() / 6 / self.module_num)
        totall_layerN = int(self.walls.__len__() / 6)
        for i in range(module_number * self.module_num, totall_layerN, 1):
            # print("module layer={}".format(i))
            dist_list_all.append(self.Distance_from_center_to_rod_by_layer(i, rodPos, rodEnd, rodRadius))
            layIndex.append(i)
        dist_list_all = np.array(dist_list_all)
        dist_list_all = dist_list_all.reshape(-1)
        # print("dist_list_all:{}".format(dist_list_all))
        minN = np.where(dist_list_all == min(dist_list_all))
        close_dist = dist_list_all[minN]
        layer_Num = layIndex[int(minN[0][0])]
        print("closest layer_Num:{}".format(layer_Num))
        dis_spring = self.Distance_from_mid_layer_to_rod_shape(layer_Num, rodPos, rodEnd, rodRadius)
        layer_Num_adj = layer_Num + module_number * self.module_num
        return [layer_Num_adj, dis_spring]

    def find_close_layer_point_by_module(self, *args, **kwargs):
        dist_list_all = []
        layIndex = []
        totall_moduleN = int(self.walls.__len__() / 6 / self.module_num)
        totall_layerN = int(self.walls.__len__() / 6)
        for i in range(10, totall_layerN, self.module_num):
            print("module layer={}".format(i))
            dist_list_all.append(self.Distance_from_mid_layer_to_target(LayerNum=i, *args, **kwargs))
            layIndex.append(i)
        dist_list_all = np.array(dist_list_all)
        dist_list_all = dist_list_all.reshape(-1)
        print("dist_list_all:{}".format(dist_list_all))
        minN = np.where(dist_list_all == min(dist_list_all))
        close_dist = dist_list_all[minN]
        layer_Num = layIndex[int(minN[0][0] / 3)]
        print("closest layer_Num:{}".format(layer_Num))
        layer_dist = (self.Distance_from_mid_layer_to_target(LayerNum=layer_Num, *args, **kwargs))
        layer_dist = np.array(layer_dist)
        maxN = np.where(layer_dist == max(layer_dist))[0][0]  # 0-2
        print("closest layer maxN:{}".format(maxN))
        # return layerNumb, maxN of spring, short dist
        return int(minN[0][0] / 3), maxN, close_dist

    def touchSense_parameters_cal(self, LayerNum, rodPos, rodEnd, rodRadius):
        spring1_pos = self.walls[(LayerNum) * 6 + 0]
        spring1_axis = self.walls[(LayerNum) * 6 + 1]
        spring1_end = spring1_pos + spring1_axis
        spring2_pos = self.walls[(LayerNum) * 6 + 2]
        spring2_axis = self.walls[(LayerNum) * 6 + 3]
        spring2_end = spring2_pos + spring2_axis
        spring3_pos = self.walls[(LayerNum) * 6 + 4]
        spring3_axis = self.walls[(LayerNum) * 6 + 5]
        spring3_end = spring3_pos + spring3_axis
        up_center, down_center = self.get_center_positions(LayerNum)
        dist_center2Rod = calulation_line_to_point(up_center, rodPos,
                                                   rodEnd)  # - rodRadius - self.spr_distance*np.sqrt(3)/2
        dist_list = self.Distance_from_mid_layer_to_rod_shape(LayerNum, rodPos, rodEnd, rodRadius)
        angle0 = triangle_angle_from_3lines(dist_center2Rod, dist_list[0], self.spr_distance[LayerNum] / np.sqrt(3))
        angle1 = triangle_angle_from_3lines(dist_center2Rod, dist_list[1], self.spr_distance[LayerNum] / np.sqrt(3))
        angle2 = triangle_angle_from_3lines(dist_center2Rod, dist_list[2], self.spr_distance[LayerNum] / np.sqrt(3))

        # angle = triangle_angle_from_3lines(dist_center2Rod, max(dist_list) + rodRadius , self.spr_distance/np.sqrt(3))
        print("touchSense_parameters_cal: angle = {}, {}, {}".format(np.rad2deg(angle0), np.rad2deg(angle1),
                                                                     np.rad2deg(angle2)))


def get_rotation_matrix_from_vector(v1, v2):
    # v1 and v2 must be unit
    v1 = v1.hat
    v2 = v2.hat
    if vector.dot(v1, v2) == 0:
        theta = pi / 2
    else:
        theta = np.arctan(vector.cross(v1, v2).mag / vector.dot(v1, v2))  # to be done??? arctan2???
    n = vector.cross(v1, v2).hat
    q0 = np.cos(theta / 2)
    q1 = np.sin(theta / 2) * n.x
    q2 = np.sin(theta / 2) * n.y
    q3 = np.sin(theta / 2) * n.z
    r11 = 2 * (q0 * q0 + q1 * q1) - 1
    r12 = 2 * (q1 * q2 - q0 * q3)
    r13 = 2 * (q1 * q3 + q0 * q2)
    r21 = 2 * (q1 * q2 + q0 * q3)
    r22 = 2 * (q0 * q0 + q2 * q2) - 1
    r23 = 2 * (q2 * q3 - q0 * q1)
    r31 = 2 * (q1 * q3 - q0 * q2)
    r32 = 2 * (q2 * q3 + q0 * q1)
    r33 = 2 * (q0 * q0 + q3 * q3) - 1
    R = [[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]]
    R = np.array(R)

    # V = [[0, -n.z, n.y],[n.z, 0, -n.x],[-n.y, n.x, 0]]
    # I = np.zeros((3, 3), int)
    # np.fill_diagonal(I, 1)
    # RR = I + V + np.matmul(V, V)/(1 + vector.dot(v1, v2))

    return R


def calulation_line_to_point(pt, v1, v2):
    # all input must be type of vector, not checking now
    a = v1 - v2
    b = pt - v2
    c = pt - v1
    a1 = np.arccos(vector.dot(a, b) / (a.mag * b.mag))
    a2 = np.arccos(vector.dot(-a, c) / (a.mag * c.mag))
    if max(np.rad2deg(a1), np.rad2deg(a2)) > 90:
        dist = min(b.mag, c.mag)
    else:
        dist = vector.cross(a, b).mag / a.mag
    return dist


def calculation_spring_vector(P0, P1, P2, P0_len, P1_len, P2_len, P_P_len):
    # P0 has the shortest len
    # P2 is the next point from conter-clockwise order, P1 is the next point from clockwise order
    # if there are two shortest len, the order should be P0_len = P2_len are shortest
    # P0, P1, P2, P0_len, P1_len, P2_len, P_P_len
    ''' trouble shooting
    P0 = self.walls[0]
    P1 = self.walls[2]
    P2 = self.walls[4]
    P0_len = 0.1
    P1_len = 0.8
    P2_len = 0.1
    P_P_len = self.spr_distance
    '''
    P0_len = round(P0_len, 7)
    P1_len = round(P1_len, 7)
    P2_len = round(P2_len, 7)
    L1 = P1_len - P0_len
    L2 = P2_len - P0_len
    # situation all len are the same
    if P1_len == P0_len and P0_len == P2_len:
        grow_dir = vector.cross(P2 - P0, P1 - P0)
        list = []
        list.append(grow_dir.hat * P0_len)
        list.append(grow_dir.hat * P1_len)
        list.append(grow_dir.hat * P2_len)
        return list
##################################################################################################
####################confidential part preparing paper later will be published #################### 
##################################################################################################
    # then translation P1 to (0,0,0) for calculation
    P0_trans = P0 - P2
    P1_trans = P1 - P2
    x, y, z = symbols('x y z')
    eqa1 = Eq(x * P0_trans.x + y * P0_trans.y + z * P0_trans.z, L2 * P_P_len * np.cos(theta))
    eqa2 = Eq(x * P1_trans.x + y * P1_trans.y + z * P1_trans.z, L2 * P_P_len * np.cos(alpha))
    eqa3 = Eq(x ** 2 + y ** 2 + z ** 2, L2 ** 2)
    sol = solve((eqa1, eqa2, eqa3), (x, y, z))
    grow_dir = vector.cross(P2 - P0, P1 - P0)
    # print("sol = {}".format(sol))
    for i in range(len(sol)):
        if not (sol[i][0].is_Float and sol[i][1].is_Float and sol[i][2].is_Float):
            print("ERROR: There is complex number from solved result {}!!!".format(sol))
            return 0
            # sys.exit("ERROR: There is complex number from solved result {}!!!". format(sol))
        vec_temp = vector(sol[i][0], sol[i][1], sol[i][2])
        if vector.dot(vec_temp, grow_dir) > 0:
            vec = vec_temp
            break

    if 'vec' not in locals():  # no vec found
        sys.exit("ERROR: There is something wrong from solved result!!!")

    P3 = vec.hat * P0_len + P0
    P4 = vec.hat * P1_len + P1
    P5 = vec.hat * P2_len + P2
    '''
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter([P0.x, P1.x, P2.x, P3.x, P4.x, P5.x],
               [P0.y, P1.y, P2.y, P3.y, P4.y, P5.y],
               [P0.z, P1.z, P2.z, P3.z, P4.z, P5.z],
               c='r', marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    '''
    list = []
    list.append(vec.hat * P0_len)
    list.append(vec.hat * P1_len)
    list.append(vec.hat * P2_len)
    # vec.hat * P0_len, vec.hat * P1_len, vec.hat * P2_len
    return list


def switch_for_cal_vec(spr1_len, spr2_len, spr3_len):
    spr1_len = round(spr1_len, 8)
    spr2_len = round(spr2_len, 8)
    spr3_len = round(spr3_len, 8)
    temp_list = []
    temp_list.append(spr1_len)
    temp_list.append(spr2_len)
    temp_list.append(spr3_len)
    Np_array = np.array(temp_list)
    min_index = np.where(Np_array == np.min(Np_array))
    switchNum = math.nan
    if min_index[0].size == 1:
        if min_index[0][0] == 0:
            switchNum = 0
        elif min_index[0][0] == 1:
            switchNum = -1
        elif min_index[0][0] == 2:
            switchNum = 1
    elif min_index[0].size == 2:
        if min_index[0][0] == 0 and min_index[0][1] == 1:
            switchNum = 0  # -1
        elif min_index[0][0] == 0 and min_index[0][1] == 2:
            switchNum = 1  # 0
        elif min_index[0][0] == 1 and min_index[0][1] == 2:
            switchNum = -1  # 1
    else:  # min_index[0].size == 3 means all len are equal
        switchNum = 0

    # print('Switch Num = {}'.format(switchNum))
    return switchNum


def swift_three_points(P1, P2, P3,rotation_angle, rot_axi):
    #function just swift the base three points of the one layer
    P_center = (P1 + P2 + P3)/3
    V1 = P1 - P_center
    V2 = P2 - P_center
    V3 = P3 - P_center
    new_V1 = rotate(V1, angle=rotation_angle, axis=rot_axi.hat) + P_center
    new_V2 = rotate(V2, angle=rotation_angle, axis=rot_axi.hat) + P_center
    new_V3 = rotate(V3, angle=rotation_angle, axis=rot_axi.hat) + P_center
    return [new_V1, new_V2, new_V3]

def Grow_toward_one_target(cells, steplenght, *args, **kwargs):
    # only apply to one cell
    # target_pos, target_radius for ball
    # rod_pos, rod_axis, rod_radius
    '''
    if len(kwargs) == 2:
        target = 'ball'
        _ball_pos = kwargs['ball_pos']
        _ball_radius = kwargs['ball_rad']
    elif len(kwargs) == 3:
        target = 'rod'
        _rod_pos = kwargs['rod_pos']
        _rod_axis = kwargs['rod_axis']
        _rod_end = _rod_pos + _rod_axis
        _rod_radius = kwargs['rod_rad']
    '''
    dis_list = cells.Distance_from_top_to_target(*args, **kwargs)
    dis_list = np.array(dis_list)
    # print('dist_list = {}'.format(dis_list))
    max_index = np.where(dis_list == max(dis_list))
    # index_grow = []  #detect grow spring num
    # index_grow.append(max_index)
    new_max_index = max_index
    while (new_max_index[0][0] == max_index[0][0]):
        res = cells.increase_one_side(steplenght, new_max_index[0][0] + 1)
        res2 = cells.increase_all_sides(steplenght * 0.5)
        dis_list = cells.Distance_from_top_to_target(*args, **kwargs)
        # dis_list = cells.Distance_from_top_to_round_shape(target_pos, target_radius)
        dis_list = np.array(dis_list)
        # print('dist_list = {}'.format(dis_list))
        new_max_index = np.where(dis_list == max(dis_list))
        if res * res2 == 0:
            return 0

    while (new_max_index[0][0] == max_index[0][0]):
        res = cells.increase_one_side(steplenght, new_max_index[0][0] + 1)
        res2 = cells.increase_all_sides(steplenght * 0.5)
        dis_list = cells.Distance_from_top_to_target(*args, **kwargs)
        # dis_list = cells.Distance_from_top_to_round_shape(target_pos, target_radius)
        dis_list = np.array(dis_list)
        # print('dist_list = {}'.format(dis_list))
        new_max_index = np.where(dis_list == max(dis_list))
        if res * res2 == 0:
            return 0

    return 1


def find_len_order_from_list(list):
    list_array = np.array(list)
    max_index = np.where(list_array == max(list_array))[0][0] + 1
    min_index = np.where(list_array == np.min(list_array))[0][0] + 1
    temp = [1, 2, 3]
    temp.remove(max_index)
    temp.remove(min_index)
    mid_index = temp[0]
    print("max_index:{0}, mid_index: {1}, min_index:{2}".format(max_index, mid_index, min_index))
    return [max_index, mid_index, min_index]


def find_len_order_from_list_from0(list):
    list_array = np.array(list)
    max_index = np.where(list_array == max(list_array))[0][0] + 1
    min_index = np.where(list_array == np.min(list_array))[0][0] + 1
    temp = [1, 2, 3]
    temp.remove(max_index)
    temp.remove(min_index)
    mid_index = temp[0]
    print("max_index:{0}, mid_index: {1}, min_index:{2}".format(max_index, mid_index, min_index))
    return [max_index - 1, mid_index - 1, min_index - 1]


def calculation_mid_point_grow_coef(list):
    list_array = np.array(list)
    print("to calculate coef: lens are {0} {1} {2} ".format(list_array[0], list_array[1], list_array[2]))
    [maxN, midN, minN] = find_len_order_from_list(list)
    Dmax = list_array[maxN - 1] - list_array[minN - 1]
    Dmid = list_array[midN - 1] - list_array[minN - 1]
    print("Dmax = {0}, Dmid = {1}, Dmid/Dmax = {2}".format(Dmax, Dmid, Dmid / Dmax))
    return Dmid / Dmax


def searching_coiling(cells, steplenght, rod_ending1, rod_ending2, rod_radius):
    layerNum, dist_spring = cells.touchSense_close_layer_point_by_module(rod_ending1, rod_ending2, rod_radius)
    [maxN, midN, minN] = find_len_order_from_list_from0(dist_spring)
    moduleNum = int(layerNum / cells.module_num)

    while (min(dist_spring) > 0.1):
        cells.increasing_one_side_by_module(steplenght * 1.5, maxN, moduleNum)
        cells.increasing_one_side_by_module(steplenght, midN, moduleNum)
        print("searching_coiling: increase springN maxN = {}, mid = {}".format(maxN, midN))
        layerNum, dist_spring = cells.touchSense_close_layer_point_by_module(rod_ending1, rod_ending2, rod_radius)
        [maxN, midN, minN] = find_len_order_from_list_from0(dist_spring)
        print("searching_coiling:dist_spring= {}, {}, {} ".format(dist_spring[0], dist_spring[1], dist_spring[2]))

    # move to next module
    total_module = int(int(cells.walls.__len__() / 6) / cells.module_num)
    pre_moduleNum = moduleNum

    for i in range(pre_moduleNum, total_module):
        new_module = i + 1
        print("new_module = {}".format(new_module))
        layerNum, dist_spring = cells.touchSense_close_layer_point_by_module_from_new(new_module, rod_ending1,
                                                                                      rod_ending2, rod_radius)
        [maxN, midN, minN] = find_len_order_from_list_from0(dist_spring)
        print("searching_coiling:dist_spring= {}, {}, {} ".format(dist_spring[0], dist_spring[1], dist_spring[2]))
        grow_adj = 0
        while (min(dist_spring) > 0.1):
            cells.increasing_one_side_by_module(steplenght * 1.5, maxN, new_module)
            cells.increasing_one_side_by_module(steplenght, midN, new_module)
            print("searching_coiling: increase springN maxN = {}, mid = {}".format(maxN, midN))
            layerNum, dist_spring = cells.touchSense_close_layer_point_by_module_from_new(new_module, rod_ending1,
                                                                                          rod_ending2, rod_radius)
            # curr_module= int(layerNum/cells.module_num)
            [maxN, midN, minN] = find_len_order_from_list_from0(dist_spring)
            print("searching_coiling:dist_spring= {}, {}, {} ".format(dist_spring[0], dist_spring[1], dist_spring[2]))
            grow_adj = 1

        while (min(dist_spring) < 0.01):
            cells.increasing_one_side_by_module(steplenght, minN, new_module)
            print("searching_coiling: increase springN min = {}".format(minN))
            layerNum, dist_spring = cells.touchSense_close_layer_point_by_module_from_new(new_module, rod_ending1,
                                                                                          rod_ending2, rod_radius)
            if grow_adj == 0:
                while (min(dist_spring) > 0.1):
                    cells.increasing_one_side_by_module(steplenght * 1.5, maxN, new_module)
                    cells.increasing_one_side_by_module(steplenght, midN, new_module)
                    print("searching_coiling: increase springN maxN = {}, mid = {}".format(maxN, midN))
                    layerNum, dist_spring = cells.touchSense_close_layer_point_by_module_from_new(new_module,
                                                                                                  rod_ending1,
                                                                                                  rod_ending2,
                                                                                                  rod_radius)
                    # curr_module= int(layerNum/cells.module_num)
                    [maxN, midN, minN] = find_len_order_from_list_from0(dist_spring)
                    print("searching_coiling:dist_spring= {}, {}, {} ".format(dist_spring[0], dist_spring[1],
                                                                              dist_spring[2]))

        print("searching_coiling for finished")

    print("searching_coiling finished")


def Grow_coiling(cells, steplenght, *args, **kwargs):
    cells.add_one_layer_on_top()
    dis_list = cells.Distance_from_top_to_target(*args, **kwargs)
    [maxN, midN, minN] = find_len_order_from_list(dis_list)

    while (1):  # np.min(dis_list) > 0.03 ):
        maxN_old = maxN
        midN_old = midN
        minN_old = minN
        while (np.min(cells.Distance_from_top_to_target(*args, **kwargs)) >= 0.2):  # too far >>> close
            print("min_dist > 0.2")
            dis_list = cells.Distance_from_top_to_target(*args, **kwargs)
            [maxN, midN, minN] = find_len_order_from_list(dis_list)
            coef = 0.5  # calculation_mid_point_grow_coef(dis_list)

            res = cells.increase_one_side(steplenght * 1, maxN)
            if res == 0:
                cells.add_one_layer_on_top()
                cells.increase_one_side(steplenght * 1, maxN)
            if np.min(cells.Distance_from_top_to_target(*args, **kwargs)) < 0.15:
                break

            res1 = cells.increase_one_side(steplenght * 0.5, midN)
            if res1 == 0:
                cells.add_one_layer_on_top()
                cells.increase_one_side(steplenght * 0.5, midN)
            if np.min(cells.Distance_from_top_to_target(*args, **kwargs)) < 0.15:
                break

            res2 = cells.increase_all_sides(steplenght * 0.2)
            if res2 == 0:
                cells.add_one_layer_on_top()
                res2 = cells.increase_all_sides(steplenght * 0.2)

        while (np.min(cells.Distance_from_top_to_target(*args, **kwargs)) >= 0.1 and np.min(
                cells.Distance_from_top_to_target(*args,
                                                  **kwargs)) < 0.2):  # max(dis_list) -  min(dis_list)> cells.spr_distance*0.85) : #to be done
            print("0.1 < min_dist < 0.2")
            dis_list = cells.Distance_from_top_to_target(*args, **kwargs)
            coef = 0.5  # calculation_mid_point_grow_coef(dis_list)
            [maxN, midN, minN] = find_len_order_from_list(dis_list)
            res = cells.increase_all_sides(stepLength=steplenght * 0.2)

            res1 = cells.increase_one_side(steplenght * 0.5, midN)
            if res1 == 0:
                cells.add_one_layer_on_top()
                cells.increase_one_side(steplenght * 0.5, midN)
            if np.min(cells.Distance_from_top_to_target(*args, **kwargs)) < 0.1:
                break

            res2 = cells.increase_one_side(steplenght * 0.5, maxN)
            if res2 == 0:
                cells.add_one_layer_on_top()
                cells.increase_one_side(steplenght * 0.5, maxN)
            if np.min(cells.Distance_from_top_to_target(*args, **kwargs)) < 0.1:
                break

            if res == 0:
                cells.add_one_layer_on_top()
                cells.increase_all_sides(steplenght * 0.2)

        while (np.min(cells.Distance_from_top_to_target(*args, **kwargs))) < 0.1:
            print("min_dist < 0.1")
            dis_list = cells.Distance_from_top_to_target(*args, **kwargs)
            [maxN, midN, minN] = find_len_order_from_list(dis_list)
            res = cells.increase_one_side(steplenght * 1, minN)
            if res == 0:
                cells.add_one_layer_on_top()
                cells.increase_one_side(steplenght * 1, minN)


def Grow_climbing(cells, steplenght, *args, **kwargs):
    '''
    list_pos = cells.get_top_ending_position()  # to be done
    z_array = np.empty(len(list_pos), dtype=object)
    for i in range(len(list_pos)):
        z_array[i] = list_pos[i].z
    '''
    # max_dist = cells.spr_distance * np.sqrt(3)/2
    cells.add_one_layer_on_top()
    dis_list = cells.Distance_from_top_to_target(*args, **kwargs)
    [maxN, midN, minN] = find_len_order_from_list(dis_list)

    while (1):  # np.min(dis_list) > 0.03 ):

        while (np.min(cells.Distance_from_top_to_target(*args, **kwargs)) >= 0.2):  # too far >>> close
            print("min_dist > 0.2")
            dis_list = cells.Distance_from_top_to_target(*args, **kwargs)
            [maxN, midN, minN] = find_len_order_from_list(dis_list)
            res = cells.increase_one_side(steplenght * 1, maxN)
            if res == 0:
                cells.add_one_layer_on_top()
                cells.increase_one_side(steplenght * 1, maxN)

            res1 = cells.increase_one_side(steplenght * 0.5, midN)
            if res1 == 0:
                cells.add_one_layer_on_top()
                cells.increase_one_side(steplenght * 0, 5, midN)

            res2 = cells.increase_all_sides(steplenght * 0.2)
            if res2 == 0:
                cells.add_one_layer_on_top()
                res2 = cells.increase_all_sides(steplenght * 0.2)

        while (np.min(cells.Distance_from_top_to_target(*args, **kwargs)) >= 0.1 and
               np.min(cells.Distance_from_top_to_target(*args, **kwargs)) < 0.2 and
               max(dis_list) - min(dis_list) > cells.spr_distance[0] * 0.85):  # to be done
            print("0.1 < min_dist < 0.2")
            dis_list = cells.Distance_from_top_to_target(*args, **kwargs)
            [maxN, midN, minN] = find_len_order_from_list(dis_list)
            res = cells.increase_all_sides(stepLength=steplenght * 0.2)
            if res == 0:
                cells.add_one_layer_on_top()
                cells.increase_all_sides(steplenght * 0.2)

            res1 = cells.increase_one_side(steplenght * 0.2, midN)
            if res1 == 0:
                cells.add_one_layer_on_top()
                cells.increase_one_side(steplenght * 0.2, midN)

            res2 = cells.increase_one_side(steplenght * 0.8, maxN)
            if res2 == 0:
                cells.add_one_layer_on_top()
                cells.increase_one_side(steplenght * 0.8, maxN)

        while (np.min(cells.Distance_from_top_to_target(*args, **kwargs))) < 0.1:
            print("min_dist < 0.1")
            dis_list = cells.Distance_from_top_to_target(*args, **kwargs)
            [maxN, midN, minN] = find_len_order_from_list(dis_list)
            cells.increase_one_side(steplenght * 1, minN)

    '''
        dis_list = cells.Distance_from_top_to_target(*args, **kwargs)
        print(dis_list)
        [maxN, midN, minN] = find_len_order_from_list(dis_list)

        res = 1
        res2 = 1
        while(res*res2 == 1 and np.min(dis_list)>0.12 ): #and max(dis_list) -  min(dis_list)> cells.spr_distance*0.9 ):
            res = cells.increase_one_side(stepLength=steplenght * 3, SprNum = maxN)
            res2 = cells.increase_one_side(stepLength=steplenght * 2, SprNum = midN)  # the 3 and 2 ration !!!!!
            dis_list = cells.Distance_from_top_to_target(*args, **kwargs)
            print(dis_list)
            [maxN, midN, minN] = find_len_order_from_list(dis_list)
    '''


def Prepare_for_grasping(cells, steplenght, *args, **kwargs):
    dis_list = cells.Distance_from_top_to_target(*args, **kwargs)
    dis_list = np.array(dis_list)
    # print('dist_list = {}'.format(dis_list))
    min_dis = np.min(dis_list)
    max_dis = np.max(dis_list)
    min_index = np.where(dis_list == min(dis_list))[0][0]
    # To be done  min_index random choose
    min_index = np.random.randint(low=0, high=3)
    min_index = cells.get_lowest_point_Num()
    # print("lowest point number spring {}".format(min_index))
    while (max_dis - min_dis < cells.spr_distance[0] * 0.85):
        res = cells.increase_one_side(steplenght * 3, min_index + 1)
        res2 = cells.increase_all_sides(steplenght * 0.5)
        if res == 0:
            cells.add_one_layer_on_top()
            cells.increase_one_side(steplenght * 3, min_index + 1)
        elif res2 == 0 and res != 0:
            cells.add_one_layer_on_top()
            cells.increase_all_sides(steplenght * 0.5)

        dis_list = cells.Distance_from_top_to_target(*args, **kwargs)
        dis_list = np.array(dis_list)
        min_dis = np.min(dis_list)
        max_dis = np.max(dis_list)
        print("elongating shortest spring. ")

    print("finished preparing ")
    return 1


def triangle_angle_from_3lines(L1, L2, L3):
    # get angle respect L3
    return np.arccos((L1 * L1 + L2 * L2 - L3 * L3) / (2 * L1 * L2))


def Show_coordinator_xyz(x, y, z):
    coordinator_org = vec(x, y, z)
    arrow_X = arrow(pos=coordinator_org, axis=vector(0.5, 0, 0), shaftwidth=0.02, color=vec(180 / 255, 180 / 255, 180 / 255))
    arrow_Y = arrow(pos=coordinator_org, axis=vector(0, 0.5, 0), shaftwidth=0.02, color=vec(180 / 255, 180 / 255, 180 / 255))
    arrow_Z = arrow(pos=coordinator_org, axis=vector(0, 0, 0.5), shaftwidth=0.02, color=vec(180 / 255, 180 / 255, 180 / 255))
    #text_x = text(pos=coordinator_org+vector(0.5, 0, 0), text='X', align='center', color=color.black)



def LightUp4():
    lamp41 = local_light(pos=vec(100, 100, -100), color=color.white * 0.4)
    lamp42 = local_light(pos=vec(100, -100, -100), color=color.white * 0.3)
    lamp41 = local_light(pos=vec(-100, 100, -100), color=color.white * 0.3)
    lamp42 = local_light(pos=vec(-100, -100, -100), color=color.white * 0.3)
    # lamp43 = local_light(pos=vec(-100,-100,100), color=color.white*0.2)
    # lamp44 = local_light(pos=vec(100,-100,100), color=color.white*0.2)
	
	
def Cal_two_vector_angle_2pi(vec1, vec2):
	#from vec2 to vec1 anticlocke wise
    #or from vec1 to vec2 clockwise
    if vector.dot(vec1.hat, vec2.hat)>1: #sometime vec1=vec2 more than 1
        return 0
    else:
        angle= np.arccos(vector.dot(vec1.hat, vec2.hat))
        sgn = vector.cross( vec2.hat, vec1.hat).z #anticlock wise is +
        return np.mod(angle * (-1)**(sgn < 0), 2 * pi)

import numpy as np


class Vpoint:
    def __init__(self) -> None:
        self.center_position3D = np.full((1,3),None) #np.zeros((1,3))
        self.act1_position3D = np.full((1,3),None) #np.zeros((1,3))
        self.act2_position3D = np.full((1,3),None) #np.zeros((1,3))
        self.act3_position3D = np.full((1,3),None) #np.zeros((1,3))

    def Center_Append(self, new_position3D:np.ndarray) -> None:
        self.center_position3D = np.vstack((self.center_position3D, new_position3D))

    def Act_Append(self, new_position3D_array:np.ndarray) -> None:
        self.act1_position3D = np.vstack((self.act1_position3D, new_position3D_array[0:3]))
        self.act2_position3D = np.vstack((self.act2_position3D, new_position3D_array[3:6]))
        self.act3_position3D = np.vstack((self.act3_position3D, new_position3D_array[6:9]))

    def Get_colomn(self, actN:int, XYZ:int)-> np.ndarray:
        if actN ==1:
            return self.act1_position3D[:,XYZ]
        elif actN ==2:
            return self.act2_position3D[:,XYZ]
        elif actN ==3:
            return self.act3_position3D[:,XYZ]


    def Get_center_colomm(self, XYZ:int)-> np.ndarray:
        return self.center_position3D[:,XYZ]


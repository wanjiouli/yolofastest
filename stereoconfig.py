import numpy as np


# 双目相机参数
class stereoCamera(object):
    def __init__(self):
        # 左相机内参
        # [[fx,0,fx],
        #  [0,fy,fy],
        #  [0,0,1]]
        self.cam_matrix_left = np.array([[1499.641, 0, 1097.616],
                                         [0., 1497.989, 772.371],
                                         [0., 0., 1.]])

        # 右相机内参
        # [[fx,0,fx],
        #  [0,fy,fy],
        #  [0,0,1]]
        self.cam_matrix_right = np.array([[1494.855, 0, 1067.321],
                                          [0., 1491.890, 777.983],
                                          [0., 0., 1.]])


        # 左右相机畸变系数:[k1, k2, p1, p2, k3]
        self.distortion_l = np.array([[-0.1103, 0.0789, -0.0004, 0.0017, -0.0095]])
        self.distortion_r = np.array([[-0.1065, 0.0793, -0.0002, -8.9263e-06, -0.0161]])

        # 旋转矩阵
        self.R = np.array([[0.9939, 0.0165, 0.1081],
                           [-0.0157, 0.9998, -0.0084],
                           [-0.1082, 0.0067, 0.9940]])

        # 平移矩阵
        self.T = np.array([[-423.716], [2.561], [21.973]])

        # 主点列坐标的差
        self.doffs = 0.0

        # 指示上述内外参是否为经过立体校正后的结果
        self.isRectified = False

    def setMiddleBurryParams(self):
        self.cam_matrix_left = np.array([[3997.684, 0, 225.0],
                                         [0., 3997.684, 187.5],
                                         [0., 0., 1.]])
        self.cam_matrix_right = np.array([[3997.684, 0, 225.0],
                                          [0., 3997.684, 187.5],
                                          [0., 0., 1.]])
        # 建立一个5行1列的零矩阵
        self.distortion_l = np.zeros(shape=(5, 1), dtype=np.float64)
        self.distortion_r = np.zeros(shape=(5, 1), dtype=np.float64)
        # 3阶方阵
        self.R = np.identity(3, dtype=np.float64)
        self.T = np.array([[-193.001], [0.0], [0.0]])
        self.doffs = 131.111
        self.isRectified = True


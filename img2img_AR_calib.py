import numpy as np
import math
import glob
import cv2


# image_path_l = r""
# image_path_r = r""
# img_name_l = glob.glob(image_path_l)
# img_name_r = glob.glob(image_path_r)
# for name_l, name_r in zip(img_name_l, img_name_r):
#     img_l = cv2.imread(name_l)
#     img_r = cv2.imread(name_r)


# p1 -> p2 transform  2D -> 2D
class SCM():
    def __init__(self, point_in, point_out):
        self.point_num = len(point_in)
        # self.pair_Num_min = (len(point_in[0]) * len(point_out[0]))
        # assert self.point_num == len(point_out) and self.point_num >= self.pair_Num_min
        self.p_in = np.array(point_in)
        self.p_out = np.array(point_out)

        self.w_in, self.h_in = self.p_in.shape
        self.w_out, self.h_out = self.p_out.shape

        self._normal()
        self._denormal()

    def _normal(self):
        self.p_in_mean = np.zeros(self.h_in)
        self.p_out_mean = np.zeros(self.h_out)

        self.p_in_scale = np.zeros(self.h_in)
        self.p_out_scale = np.zeros(self.h_out)

        for i in range(self.point_num):
            self.p_in_mean += self.p_in[i]
            w = []
            for j in range(self.h_in):
                w.append(self.p_in[i, j] ** 2)
            w = np.array(w)
            self.p_in_scale += w

            t = []
            self.p_out_mean += self.p_out[i]
            for j in range(self.h_out):
                t.append(self.p_out[i, j] ** 2)
            t = np.array(t)
            self.p_out_scale += t

        self.p_in_mean /= self.point_num
        self.p_in_scale /= self.point_num
        self.p_out_mean /= self.point_num
        self.p_out_scale /= self.point_num

        for i in range(self.h_in):
            self.p_in_scale[i] = math.sqrt(self.p_in_scale[i] - (self.p_in_mean[i] ** 2))
        for i in range(self.h_out):
            self.p_out_scale[i] = math.sqrt(self.p_out_scale[i] - (self.p_out_mean[i] ** 2))
        return 1

    def _denormal(self):
        self.p_in_nom = np.zeros((3, 3))
        self.p_out_nom = np.zeros((3, 3))
        self.p_in_nom[-1, -1], self.p_out_nom[-1, -1] = 1., 1.

        for i in range(self.h_in):
            self.p_in_nom[i, i] = 1 / self.p_in_scale[i]
            self.p_in_nom[i, self.h_in] = -self.p_in_mean[i] * self.p_in_nom[i, i]
        for i in range(self.h_out):
            self.p_out_nom[i, i] = self.p_out_scale[i]
            self.p_out_nom[i, self.h_out] = self.p_out_mean[i]

    def get_matrix_of_p2p(self):
        A = np.zeros((2 * self.point_num, 9))
        for i in range(self.point_num):
            # 将数据归一化
            p_in__ = np.divide(self.p_in[i] - self.p_in_mean, self.p_in_scale)
            p_out__ = np.divide(self.p_out[i] - self.p_out_mean, self.p_out_scale)
            # 源数据处理
            # p_in__ = self.p_in[i]
            # p_out__ = self.p_out[i]

            A[i * 2][0] = p_in__[0]
            A[i * 2][1] = p_in__[1]
            A[i * 2][2] = 1
            A[i * 2][3] = 0
            A[i * 2][4] = 0
            A[i * 2][5] = 0
            A[i * 2][6] = -p_in__[0] * p_out__[0]
            A[i * 2][7] = -p_in__[1] * p_out__[0]
            A[i * 2][8] = -p_out__[0]

            A[i * 2 + 1][0] = 0
            A[i * 2 + 1][1] = 0
            A[i * 2 + 1][2] = 0
            A[i * 2 + 1][3] = p_in__[0]
            A[i * 2 + 1][4] = p_in__[1]
            A[i * 2 + 1][5] = 1
            A[i * 2 + 1][6] = -p_in__[0] * p_out__[1]
            A[i * 2 + 1][7] = -p_in__[1] * p_out__[1]
            A[i * 2 + 1][8] = -p_out__[1]

        # 奇异值分解 uu
        uu, ss, V = np.linalg.svd(A)
        V = V.transpose()

        # 最小奇异向量
        smallestValues = V[:, -1].reshape(3, 3)

        self.G = np.dot(np.dot(self.p_out_nom, smallestValues), self.p_in_nom)

        normalDirection = math.sqrt(self.G[2, 0] ** 2 + self.G[2, 1] ** 2 + self.G[2, 2] ** 2)

        self.G *= 1 / normalDirection

        isNegative = self.G[2, 0] * self.p_in[1, 0] + self.G[2, 1] * self.p_in[1, 1] + self.G[2, 2]

        if isNegative < 0:
            self.G *= -1

        return self.G


# -*- python coding: utf-8 -*-
# @Time: 6月 06, 2022
# ---
import cv2
import numpy as np
from little_function import *

LEFT_2_RIGHT = 0
RIGHT_2_LEFT = 1


def get_region_depth(frame_src, frame_dst, map_object):
    """
    :param frame_src:
    :param frame_dst:
    :param map_object:
    :return:
    """
    depth = [4000]  # default value
    region = np.zeros_like(frame_src)
    size = 60
    try:
        #  The current code does not guarantee that the two sets of points correspond to each other,
        #  So we must ensure that there is only one circle in the image
        p_src, _ = gen_circle_center(frame_src, "c_src")
        p_dst, _ = gen_circle_center(frame_dst, "c_dst")
    except:
        # The reason for the inaccuracy of finding the circle is the change of light and shadow
        print("There is no circle in image!")
        p_src = p_dst = None

    # 根据点对，计算出深度：即目标离相机的距离
    try:
        if len(p_src) == len(p_dst) and len(p_src) != 0 and len(p_dst) != 0:
            if map_object.flag == 0:
                distance = map_object.get_depth(p_src, p_dst)
            else:
                distance = map_object.get_depth(p_dst, p_src)
            # print("The depth is", distance[0])
            region[p_src[:, 1][0] - size:p_src[:, 1][0] + size, p_src[:, 0][0] - size:p_src[:, 0][0] + size] = 255
            depth = distance
            x, y = map_object.compute_error(p_src, p_dst, depth)
            print(x, y)
            # 深度误差与融合误差分析
            for depth_err in range(-1000, 3000, 100):
                depth_wrong = depth + depth_err
                x_wrong, y_wrong = map_object.compute_error(p_src, p_dst, depth_wrong)
                # print(x_wrong, y_wrong, depth_err)
        else:
            print("没找到目标！")
            depth = [3500]
    except:
        print("未知错误！")

    return region, depth


class ImageMapping:
    def __init__(self, Kl=None, Kr=None, R=None, T=None, flag=0, G=None):
        """
        :param Kl: 左相机内参
        :param Kr: 右相机内参
        :param R: 左相机->右相机旋转
        :param T: 左相机->右相机平移
        :param flag: 0:left->right  1:right->left
        """
        self.Kl = Kl
        self.Kr = Kr
        self.R = R
        self.T = T
        self.flag = flag
        self.G = G
        self.RT = np.concatenate((self.R, self.T), axis=1)

    def image_align(self, image_src, image_dst, region=None):
        """
        Image fidelity expansion based on transform size
        :param image_src:
        :param image_dst:
        :param region:
        :return:
        """
        if region is None:
            region = np.full_like(image_src, 255)
        h_d, w_d = image_dst.shape[:2]
        h_s, w_s = image_src.shape[:2]

        wh_rate = w_d / h_d  # 保真映射长宽比
        new_h = h_s
        new_w = round(h_s * wh_rate)
        if new_w - w_s > 0:
            new_src = np.concatenate((np.zeros((new_h, new_w - w_s)).astype('uint8'), image_src), axis=1)
            new_region = np.concatenate((np.zeros((new_h, new_w - w_s)).astype('uint8'), region), axis=1)
        else:
            new_src = image_src[:, w_s - new_w:]
            new_region = region[:, w_s - new_w:]

        rate = [new_h / h_d, new_w / w_d]  # (h_rate, w_rate) ---scale
        trans = [new_h - h_s, new_w - w_s]  # 偏移量

        # new Kl  new Kr
        if self.flag == 0:
            new_Kl = self.Kl.copy()
            new_Kl[0][2] += trans[1]
            new_Kl[0][1] += trans[0]
            new_Kr = self.Kr.copy()
            rate_arr = np.expand_dims(np.append(rate, 1), axis=1)
            rate_arr = np.concatenate((rate_arr, rate_arr, rate_arr), axis=1)
            new_Kr = np.multiply(new_Kr, rate_arr)
        elif self.flag == 1:
            new_Kl = self.Kl.copy()
            rate_arr = np.expand_dims(np.append(rate, 1), axis=1)
            rate_arr = np.concatenate((rate_arr, rate_arr, rate_arr), axis=1)
            new_Kl = np.multiply(new_Kl, rate_arr)
            new_Kr = self.Kr.copy()
            new_Kr[0][2] += trans[1]
            new_Kr[0][1] += trans[0]
        else:
            new_src = image_src.copy()
            new_region = region.copy()
            new_Kl = self.Kl.copy()
            # new_Kl[0][2] += trans[1]
            # new_Kl[0][1] += trans[0]
            new_Kr = self.Kr.copy()
            rate_arr = np.expand_dims(np.append(rate, 1), axis=1)
            rate_arr = np.concatenate((rate_arr, rate_arr, rate_arr), axis=1)
            new_Kr = np.multiply(new_Kr, rate_arr)

        return new_src, new_region, new_Kr, new_Kl

    @classmethod
    def xy_trans(cls, point, _Kl, Kr, RT, depth, size=(1280, 1024)):
        """
        left -> right
        :param point: 图像坐标,[[x,y],[],[]]
        :param _Kl: 左相机内参矩阵的逆矩阵
        :param Kr: 右相机内参矩阵的逆矩阵
        :param RT: 右相机相对左相机的RT矩阵即 Pl = R Pr + T
        :param depth: point的世界坐标深度
        :param size: 图像的尺寸（w,h）
        :return: mask:转换后图像坐标是否还在图像范围内，result图像坐标[[x,y,1],[]]
        """
        xy1 = np.concatenate((point, np.ones((point.shape[0], 1))), axis=1)  # [[x,y,1],[]]
        xy1 = np.transpose(xy1, (1, 0))  # [[x...],[y...],[1...]]
        xyd = _Kl.dot(xy1) * depth  # [[x...],[y...],[d...]]
        xyd1 = np.concatenate((xyd, np.ones((1, xyd.shape[1]))), axis=0)  # [[x...],[y...],[d...],[1...]]
        xyz = Kr.dot(RT.dot(xyd1))
        xyz = np.transpose(xyz, (1, 0))
        result_xyz = xyz / xyz[:, 2:]
        out = np.round(result_xyz).astype(np.int32)
        mask = np.array(out[:, 0] > 0) * np.array(out[:, 0] < size[0]) * \
               np.array(out[:, 1] > 0) * np.array(out[:, 1] < size[1])
        result = out[mask]
        return mask, result

    @classmethod
    def xy_trans_rl(cls, point, Kl, _Kr, R, T, depth, size=(1280, 1024)):
        """
        right -> left
        :param point:图像坐标,[[x,y],[],[]]
        :param Kl: 左相机内参矩阵的逆矩阵
        :param _Kr:
        :param R:
        :param T: 右相机相对左相机的RT矩阵即 Pr = R^-1(Pl - T)
        :param depth: point的世界坐标深度
        :param size: 目标图像的尺寸（w,h）
        :return: mask:转换后图像坐标是否还在图像范围内，result图像坐标[[x,y,1],[]]
        """
        xy1 = np.concatenate((point, np.ones((point.shape[0], 1))), axis=1)  # [[x,y,1],[]]
        xy1 = np.transpose(xy1, (1, 0))  # [[x...],[y...],[1...]]
        # xyz = Kl.dot(np.linalg.inv(R).dot(_Kr.dot(xy1) * depth - T))
        xyd = _Kr.dot(xy1) * depth  # [[x...],[y...],[d...]]
        xyz = Kl.dot(np.linalg.inv(R).dot(xyd - T))  # [[x...],[y...],[z...]]
        xyz = np.transpose(xyz, (1, 0))
        result_xyz = xyz / xyz[:, 2:]
        out = np.round(result_xyz).astype(np.int32)
        mask = np.array(out[:, 0] > 0) * np.array(out[:, 0] < size[0]) * \
               np.array(out[:, 1] > 0) * np.array(out[:, 1] < size[1])
        result = out[mask]
        return mask, result

    def compute_error(self, point_src, point_dst, depth, size=(1280, 1024)):
        """
        :param point_src:
        :param point_dst:
        :param depth:
        :param size:
        :return:
        """
        assert len(point_src) == len(point_dst), "the points num is unequal!"
        src_xy = np.array(point_src).reshape(-1, 2)
        if self.flag == 0:
            _, dst_xyz = self.xy_trans(src_xy, np.linalg.inv(self.Kl), self.Kr, self.RT, depth, size)
        else:
            _, dst_xyz = self.xy_trans_rl(src_xy, self.Kl, np.linalg.inv(self.Kr), self.R, self.T, depth, size)

        if dst_xyz.size == 0:
            return [], []
        else:
            out = point_dst - dst_xyz[:, 0:2]
            x_error, y_error = out[0]
            return x_error, y_error

    def get_depth(self, point_left, point_right):
        """
        通过相机对应点，获取深度
        :param point_left: 左相机图像坐标[[x,y],[]]
        :param point_right: 右相机图像坐标
        :return: 估计深度
        """
        _R = np.linalg.inv(self.R)
        _k1 = np.linalg.inv(self.Kl)
        _k2 = np.linalg.inv(self.Kr)
        xy = np.array(point_left).reshape(-1, 2)
        xy1 = np.concatenate((xy, np.ones((xy.shape[0], 1))), axis=1).T
        uv = np.array(point_right).reshape(-1, 2)
        uv1 = np.concatenate((uv, np.ones((uv.shape[0], 1))), axis=1).T
        xyz = _k1.dot(xy1)
        uvw = _k2.dot(uv1)
        result_ = self.R.dot(xyz) - uvw
        out = np.divide(self.T, result_)
        # the value of axis=x is most correct
        depth = np.abs(np.round(out[0, :]))
        return depth

    def fusion(self, image_src, image_dst, region=None, depth=0):
        """
        :param image_src: 含有目标的图像
        :param image_dst: 结果图像，image_src映射到此图像
        :param region: 融合的区域
        :param depth: 区域的深度
        :return:
        """
        if region is None:
            region = np.full_like(image_src, 255)
        new_src, new_region, new_Kr, new_Kl = self.image_align(image_src, image_dst, region)

        _new_Kl = np.linalg.inv(new_Kl)
        h, w = new_src.shape[:2]

        yx_input = np.argwhere(new_region > 0)
        if yx_input.size == 0:
            new_region = np.full_like(new_src, 255)
            yx_input = np.argwhere(new_region > 0)
        xy_input = yx_input[:, [1, 0]]
        if self.flag == 0:
            mask, result_ = self.xy_trans(xy_input, _new_Kl, new_Kr, self.RT, depth, (w, h))
        else:
            mask, result_ = self.xy_trans_rl(xy_input, new_Kl, np.linalg.inv(new_Kr), self.R, self.T, depth, (w, h))

        src_ = xy_input[mask]
        img = np.zeros_like(new_src)
        img[(result_[:, 1], result_[:, 0])] = new_src[(src_[:, 1], src_[:, 0])]
        cv2.imshow("tttt", img)  # 马赛克需要插值
        img = cv2.resize(img, (image_dst.shape[1], image_dst.shape[0]), interpolation=cv2.INTER_LINEAR)
        cv2.imshow("trans", img)
        out = image_dst.copy()
        out = cv2.addWeighted(out, 0.5, img, 0.5, 0)

        FPGA = False
        if FPGA is True:
            G = new_Kr.dot(self.RT)
            G[:, 3:] = G[:, 3:] / depth
            kuozhan = np.ones((xy_input.shape[0], 1))
            zhongjian1 = np.concatenate((xy_input, kuozhan), axis=1).T
            zhongjian2 = _new_Kl.dot(zhongjian1)
            zhongjian3 = np.concatenate((zhongjian2, np.ones((1, zhongjian2.shape[1]))), axis=0)
            # G0_ = G[0:1, :]
            # zhongjian3_ = zhongjian3.T
            # zhongjian_shuchu = np.multiply(G0_, zhongjian3_)

            shuchu = G.dot(zhongjian3).T

            # shuchu = np.round(shuchu).astype(np.int32)
            # mmmmmm = np.array(shuchu[:, 0] > 0) * np.array(shuchu[:, 0] < 640) * \
            #          np.array(shuchu[:, 1] > 0) * np.array(shuchu[:, 1] < 512)
            # shuchu_test = shuchu[mmmmmm]
            # src_test = xy_input[mmmmmm]
            print(G)

            M_l = _new_Kl * (2 ** 30)
            G_l = G * (2 ** 20)

            # txt_path_1 = "../txt_file/calib_.txt"
            # calib_txt = open(txt_path_1, "w")
            # for i in range(len(M_l)):
            #     calib_txt.write(str(M_l[i]) + '\n')
            # calib_txt.write('\n\n\n')
            # for i in range(len(G_l)):
            #     calib_txt.write(str(G_l[i]) + '\n')
            # calib_txt.close()
            #
            # xy_M_path = "../txt_file/xy_M.txt"
            # xy_M = open(xy_M_path, "w")
            # Y_OUT = zhongjian2.T
            # for i in range(len(Y_OUT)):
            #     xy_M.write(str(Y_OUT[i]) + '\n')
            # xy_M.close()
            #
            # xy_dst_path = "../txt_file/xy_dst.txt"
            # xy_dst = open(xy_dst_path, "w")
            # for i in range(len(shuchu)):
            #     xy_dst.write(str(shuchu[i]) + '\n')
            # xy_dst.close()
            cv2.waitKey()

        return out

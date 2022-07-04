# _*_ coding: utf-8 _*_
"""
Time:     2022/6/30 15:53
Author:   bwang
File:     inv_Image_Mapping.py
"""
import cv2
import numpy as np

from fusion_function import *


class ImageFusionInv:
    """
    图像融合算法，包含反映射方法
    """

    def __init__(self, Kl=None, Kr=None, R=None, T=None, dst_size=(1024, 1280)):
        """
        :param Kl: 左相机内参
        :param Kr: 右相机内参
        :param R: 左相机->右相机旋转
        :param T: 左相机->右相机平移
        """
        self.Kl = Kl
        self.Kr = Kr
        self.R = R
        self.T = T
        self.RT = np.concatenate((self.R, self.T), axis=1)
        self.dst_size = dst_size
        self.wh_rate = dst_size[1] / dst_size[0]

    def get_trans_param(self, src_size: tuple, dst_size: tuple, depth):
        """
        :param src_size: 变换前尺寸
        :param dst_size: 变换后尺寸
        :param depth: 目标深度
        :return: 做相机内参逆矩阵， Kr*RT
        """
        h_dst, w_dst = dst_size
        h_src, w_src = src_size
        wh_rate = w_dst / h_dst  # 保真映射长宽比
        new_h = h_src
        new_w = myround(h_src * wh_rate)
        trans = [new_h - h_src, new_w - w_src]  # 偏移量

        new_Kl = self.Kl.copy()
        new_Kl[0][2] += trans[1]
        new_Kl[0][1] += trans[0]
        _new_Kl = np.linalg.inv(new_Kl)

        src_h, src_w = src_size
        rate = [src_w / w_dst, src_h / h_dst]

        M12 = self.Kr.dot(self.RT)  # 矩阵运算的结合律
        # 将其转换到相同的对比度下
        M12[0:1, :] = M12[0:1, :] * rate[0]
        M12[1:2, :] = M12[1:2, :] * rate[1]
        M12[:, 3:] = M12[:, 3:] / depth

        return _new_Kl, M12

    @classmethod
    def xy_trans(cls, region, _Kl, M12, dst_size=(512, 640)):
        """
        :param region:
        :param _Kl: 做相机内参矩阵逆矩阵
        :param M12: RT隐去深度
        :param dst_size: 图像的尺寸（w,h）
        :return: src_xy, result_xy, 初始target_rect = (0, 0, 0, 0)
        """
        target_rect = (0, 0, 0, 0)

        point_yx = np.argwhere(region)
        point = point_yx[:, [1, 0]]  # 图像坐标,[[x,y],[],[]]

        xy1 = np.concatenate((point, np.ones((point.shape[0], 1))), axis=1)  # [[x,y,1],[]]
        xy1 = np.transpose(xy1, (1, 0))  # [[x...],[y...],[1...]]
        xyd = _Kl.dot(xy1)  # [[x...],[y...],[d...]]
        xyd1 = np.concatenate((xyd, np.ones((1, xyd.shape[1]))), axis=0)  # [[x...],[y...],[d...],[1...]]
        xyz = M12.dot(xyd1)
        xyz = np.transpose(xyz, (1, 0))
        result_xyz = xyz / xyz[:, 2:]
        out = myround(result_xyz).astype(np.int32)

        mask = np.array(out[:, 0] >= 0) * np.array(out[:, 0] < dst_size[0]) * np.array(out[:, 1] >= 0) * np.array(
            out[:, 1] < dst_size[1])
        result_xy = out[mask]
        src_xy = point[mask]

        rect_mask = np.zeros_like(region).astype("uint8")
        rect_mask[result_xy[:, 1], result_xy[:, 0]] = 255
        contours, hierarchy = cv2.findContours(rect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        assert contours is not None, "There is no contours!"
        for cont in contours:
            # 外接矩形
            target_rect = cv2.boundingRect(cont)

        return src_xy, result_xy, target_rect

    def image_fusion(self, image_src: np.array, target_mask: np.array, image_dst: np.array, depth=0):
        """
        :param image_src: 目标源图
        :param target_mask:目标框位置
        :param image_dst:目标映射图
        :param depth:目标深度
        :return:
        """

        if target_mask is None:  # 没有目标，不映射
            target_mask = np.full_like(image_src, 255)
        h_dst, w_dst = image_dst.shape[:2]

        inv_kl, g12 = self.get_trans_param(image_src.shape[:2], image_dst.shape[:2], depth)
        xy_src, xy_result, rect = self.xy_trans(target_mask, inv_kl, g12, (h_dst, w_dst))

        x, y, w, h = rect
        out_target = image_dst.copy()
        out_target[y:y + h, x:x + w] = 0
        out_target[xy_result[:, 1], xy_result[:, 0]] = image_src[xy_src[:, 1], xy_src[:, 0]]
        out_mf = maxblur(out_target, rect)

        out = cv2.resize(out_mf, tuple(reversed(image_dst.shape[:2])), interpolation=cv2.INTER_LINEAR)

        # 反向计算
        # inv_g33 = self.get_inv_param(g12)
        # out_1 = self.inv_image_fusion(image_src, rect, inv_g33)
        # out_1 = cv2.resize(out_1, (1280, 1024), interpolation=cv2.INTER_LINEAR)
        # out_1 = cv2.addWeighted(out_1, 0.5, image_dst, 0.5, 0)

        return out

    def get_inv_param(self, g12):

        g33 = np.array([
            [g12[0, 0], g12[0, 1], g12[0, 2] + g12[0, 3]],
            [g12[1, 0], g12[1, 1], g12[1, 2] + g12[1, 3]],
            [0, 0, 1]
        ])
        inv_g33 = self.Kl.dot(np.linalg.inv(g33))
        return inv_g33

    @classmethod
    def inv_image_fusion(cls, image_src, rect, inv_g33):

        rect_x, rect_y, rect_w, rect_h = rect

        tar_img = np.zeros_like(image_src)
        tar_img[rect_y:rect_y + rect_h, rect_x:rect_x + rect_w] = 255
        tar_yx = np.argwhere(tar_img > 0)
        tar_xy = tar_yx[:, [1, 0]]
        tar_xy1 = np.concatenate((tar_xy, np.ones((tar_xy.shape[0], 1))), axis=1)
        out_xy1 = inv_g33.dot(tar_xy1.T)
        out_xy1 = myround(out_xy1).astype(np.int32)

        tar_img[tar_xy[:, 1], tar_xy[:, 0]] = image_src[out_xy1[1, :], out_xy1[0, :]]
        tar_img = tar_img.astype("uint8")

        return tar_img

# -*- python coding: utf-8 -*-
# @Time: 4月 28, 2022
# ---

import numpy as np
import cv2 as cv
from glob import glob
import os


def get_object_points(m, n, k):
    # 计算真实坐标 # 获取标定板坐标角点坐标（世界坐标系
    # 参数：内点行数，内点列， 标定板大小
    # 返回值：数组，（m*n行，3列），真实内点坐标
    obj_point = np.zeros(shape=(m * n, 3), dtype=np.float32)
    for i in range(m * n):
        obj_point[i][0] = i % m
        obj_point[i][1] = int(i / m)
    return obj_point * k


def main():
    # 相机标定参数设定（单目，双目）
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    criteria_stereo = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # 计算标定板真实坐标，标定板内点9*6，大小10mm*10mm
    obj_point = get_object_points(9, 6, 10)

    obj_points = []
    points_l = []
    points_r = []

    for i in range(len(image_list_l)):
        # 分别读取每张图片并转化为灰度图
        img_l = cv.imread(image_list_l[i])
        img_r = cv.imread(image_list_r[i])
        gray_l = cv.cvtColor(img_l, cv.COLOR_BGR2GRAY)
        gray_r = cv.cvtColor(img_r, cv.COLOR_BGR2GRAY)
        # opencv寻找角点
        ret_l, corners_l = cv.findChessboardCorners(gray_l, (9, 6), None)
        ret_r, corners_r = cv.findChessboardCorners(gray_r, (9, 6), None)
        if (ret_l & ret_r) is True:
            # opencv对真实坐标格式要求，vector<vector<Point3f>>类型
            obj_points.append(obj_point)
            # 角点细化
            corners_ll = cv.cornerSubPix(gray_l, corners_l, (5, 5), (-1, -1), criteria)
            corners_rr = cv.cornerSubPix(gray_r, corners_r, (5, 5), (-1, -1), criteria)
            points_l.append(corners_ll)
            points_r.append(corners_rr)
    # 对左右相机分别进行单目相机标定（复制时格式可能有点问题，用pycharm自动格式化）
    ret_ll, camera_matrix_l, dist_l, R_l, T_l = cv.calibrateCamera(obj_points, points_l, (640, 480), None, None)
    ret_rr, camera_matrix_r, dist_r, R_l, T_r = cv.calibrateCamera(obj_points, points_r, (640, 480), None, None)
    # 双目相机校正
    # ret_s, mLS, dLS, mRS, dRS, R, T, E, F = cv.stereoCalibrate(obj_points, points_l,
    #                                                            points_r, camera_matrix_l,
    #                                                            dist_l, camera_matrix_r,
    #                                                            dist_r, (640, 480),
    #                                                            criteria_stereo, flags=cv.CALIB_USE_INTRINSIC_GUESS)
    ret_s, camera1, dist1, camera2, dist2, R, T, E, F = cv.stereoCalibrate(obj_points, points_l,
                                                                           points_r, camera_matrix_l,
                                                                           dist_l, camera_matrix_r,
                                                                           dist_r, (640, 480),
                                                                           criteria_stereo,
                                                                           flags=cv.CALIB_USE_INTRINSIC_GUESS)
    # 标定结束，结果输出，cameraMatrixL，cameraMatrixR分别为左右相机内参数矩阵
    # R， T为相机2与相机1旋转平移矩阵
    print(camera_matrix_l)
    print('*' * 20)
    print(camera_matrix_r)
    print('*' * 20)
    print(R)
    print('*' * 20)
    print(T)


if __name__ == "__main__":
    image_path_l = r""
    image_path_r = r""
    image_list_l = glob(image_path_l)
    image_list_r = glob(image_path_r)
    main()

# -*- python coding: utf-8 -*-
# @Time: 5月 23, 2022
# ---
import os
import cv2
import numpy as np


def getImageList(img_dir):
    # 获取图片文件夹位置，方便opencv读取
    # 参数：照片文件路径
    # 返回值：数组，每一个元素表示一张照片的绝对路径
    imgPath = []
    if os.path.exists(img_dir) is False:
        print('error dir')
    else:
        for parent, dirNames, fileNames in os.walk(img_dir):
            for fileName in fileNames:
                imgPath.append(os.path.join(parent, fileName))
    return imgPath


def getObjectPoints(m, n, k):
    # 计算真实坐标
    # 参数：内点行数，内点列， 标定板大小
    # 返回值：数组，（m*n行，3列），真实内点坐标
    objP = np.zeros(shape=(m * n, 3), dtype=np.float32)
    for i in range(m * n):
        objP[i][0] = i % m
        objP[i][1] = int(i / m)
    return objP * k


# 相机标定参数设定（单目，双目）
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# 计算标定板真实坐标，标定板内点9*6，大小10mm*10mm
objPoint = getObjectPoints(8, 11, 45)

objPoints = []
imgPointsL = []
imgPointsR = []
# 相片路径，自行修改
imgPathL = r'../calib2/1'
imgPathR = r'../calib2/2'
filePathL = getImageList(imgPathL)
filePathR = getImageList(imgPathR)

for i in range(len(filePathL)):
    # 分别读取每张图片并转化为灰度图
    imgL = cv2.imread(filePathL[i])
    imgR = cv2.imread(filePathR[i])
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    cv2.imshow("L", grayL)
    cv2.imshow("R", grayR)
    cv2.waitKey(1)
    # opencv寻找角点
    retL, cornersL = cv2.findChessboardCorners(grayL, (8, 11), flags=2)

    retR, cornersR = cv2.findChessboardCorners(grayR, (8, 11), flags=2)
    if (retL & retR) is True:
        # opencv对真实坐标格式要求，vector<vector<Point3f>>类型
        objPoints.append(objPoint)
        # 角点细化
        cornersL2 = cv2.cornerSubPix(grayL, cornersL, (3, 3), (-1, -1), criteria)
        cornersR2 = cv2.cornerSubPix(grayR, cornersR, (3, 3), (-1, -1), criteria)
        imgPointsL.append(cornersL2)
        imgPointsR.append(cornersR2)


if objPoints:
    print(len(objPoints))
    # 对左右相机分别进行单目相机标定（复制时格式可能有点问题，用pycharm自动格式化）
    retL, cameraMatrixL, distMatrixL, RL, TL = cv2.calibrateCamera(objPoints, imgPointsL, (1280, 1024), None, None)
    retR, cameraMatrixR, distMatrixR, RR, TR = cv2.calibrateCamera(objPoints, imgPointsR, (1280, 1024), None, None)
    # 双目相机校正 CALIB_USE_INTRINSIC_GUESS  flag使用CALIB_FIX_INTRINSIC，解决大小不同分辨率的标定问题
    retS, mLS, dLS, mRS, dRS, R, T, E, F = cv2.stereoCalibrate(objPoints, imgPointsL,
                                                               imgPointsR, cameraMatrixL,
                                                               distMatrixL, cameraMatrixR,
                                                               distMatrixR, (1280, 1024),
                                                               criteria_stereo, flags=cv2.CALIB_USE_INTRINSIC_GUESS)
    # 标定结束，结果输出，cameraMatrixL，cameraMatrixR分别为左右相机内参数矩阵
    # R， T为相机2与相机1旋转平移矩阵
    print(cameraMatrixL)
    print('*' * 20)
    print(cameraMatrixR)
    print('*' * 20)
    print(R)
    print('*' * 20)
    print(T)

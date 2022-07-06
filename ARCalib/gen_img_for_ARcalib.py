# -*- coding: utf-8 -*-
# @Time    : 2022/6/13 15:39
# @Author  : Daniel Zhang
# @Email   : zhangdan_nuaa@163.com
# @File    : gen_img.py
# @Software: PyCharm
"""
本文件实现AR的标定，图像和AR设备采点
步骤一：带上HMD，运行本文件，你将在AR上看到十字图标若干
步骤二：将现实的标志点固定在距离AR位置的深度depth处
步骤三：将AR屏幕上的十字图标，与现实的标定点对齐，鼠标在AR屏幕上点击鼠标左键拍图，通过转动头，对齐所有的点并拍照
步骤四：将depth进行改变，并改变保存文件的路径
"""

from little_function import *
import numpy as np
import cv2
import os

num = 0
point_circle = []
point_press = []


def mouse(event, x, y, flag, param):
    frame = param[0]
    path = param[1]
    depth = param[2]
    global num
    if event == cv2.EVENT_LBUTTONDOWN:
        print("------")
        xy = "%d %d %d" % (x, y, depth)
        point_press.append(xy)
        print(xy)
        num += 1
        _, frame_binary = cv2.threshold(frame, 180, 255, cv2.THRESH_BINARY_INV)
        cicle_center, radius = gen_circle_center(frame_binary, b_display=True)
        cv2.imwrite(path + "{:0=8}.jpg".format(num), frame)
        assert len(cicle_center) != 0
        for i in range(len(cicle_center)):
            c_xy = "%d %d %d" % (cicle_center[0][0], cicle_center[0][1], depth)
            point_circle.append(c_xy)
            print(c_xy)


def cross_frame(frame: np.array, line_h_number=3, line_w_number=3, color=(255, 0, 0)):
    """
    给图像frame画十字点
    :param frame:
    :param line_h_number:
    :param line_w_number:
    :param color:
    :return:
    """
    cross_xy = []
    step_h = int(frame.shape[0] / line_h_number)
    step_w = int(frame.shape[1] / line_w_number)
    for j in range(int(step_h / 2), frame.shape[0], step_h):
        for i in range(int(step_w / 2), frame.shape[1], step_w):
            frame = cv2.line(frame, (i - 40, j), (i + 40, j), color, 4)
            frame = cv2.line(frame, (i, j - 40), (i, j + 40), color, 4)
            frame = cv2.circle(frame, (i, j), 7, color, 7)
            p_xy = "%d %d" % (i, j)
            cross_xy.append(p_xy)
            print(i, j)
    return frame, cross_xy


def main():
    temp_frame = np.zeros((1080, 1920))
    cv2.line(temp_frame, (0, 0), (1919, 0), (255, 0, 0), 1)
    cv2.line(temp_frame, (0, 0), (0, 1079), (255, 0, 0), 1)
    cv2.line(temp_frame, (0, 1079), (1919, 1079), (255, 0, 0), 1)
    cv2.line(temp_frame, (1919, 0), (1919, 1079), (255, 0, 0), 1)
    temp_frame, cross_points = cross_frame(temp_frame, 4, 4, (255, 0, 0))

    cap = cv2.VideoCapture(0)
    #  重要的深度设计
    depth = 5000  # 请输入数据采集深度

    now_path = "../image_calib/image_{}/".format(depth)
    if not os.path.exists(now_path):
        os.mkdir(now_path)

    while True:
        rt, frame = cap.read()
        assert rt is True, "find no camera !"
        cv2.imshow("camera", frame)
        cv2.namedWindow('binary_left', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("binary_left", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.moveWindow("binary_left", 2560, 0)
        cv2.imshow("binary_left", temp_frame)
        cv2.setMouseCallback("binary_left", mouse, [frame, now_path, depth])
        key = cv2.waitKey(30)

        if key == 27:
            txt = open(now_path + "point_ar_cross.txt", "w")
            for line in cross_points:
                txt.write(line + '\n')
            txt.close()
            txt = open(now_path + "point_3d_circle.txt", "w")
            for line in point_circle:
                txt.write(line + '\n')
            txt.close()
            txt = open(now_path + "point_3d_press.txt", "w")
            for line in point_press:
                txt.write(line + '\n')
            txt.close()
            exit()


if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()

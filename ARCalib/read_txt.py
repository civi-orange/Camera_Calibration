# -*- coding: utf-8 -*-
# @Time    : 2022/6/13 17:16
# @Author  : Daniel Zhang
# @Email   : zhangdan_nuaa@163.com
# @File    : read_txt.py
# @Software: PyCharm
import cv2
import numpy as np


def read_txt(filepath, col=2):
    zz = []
    ff = open(filepath, "r")
    ftxt = ff.readlines()
    for fline in ftxt:
        fline = fline.strip()
        for ii in range(len(fline.split())):
            fx = fline.split()[ii]
            zz.append(int(fx))
    p = np.array(zz).reshape(-1, col)
    return p


if __name__ == '__main__':

    z = []
    f = open("./image_out1.txt", "r")
    txt = f.readlines()
    i = 0
    for line in txt:
        i += 1
        line = line.strip()
        for i in range(len(line.split())):
            x = line.split()[i]
            x = int(x, 16)
            z.append(x)
    print(len(z))
    zzz = np.array(z).astype("uint8")
    if len(zzz) < 104918:
        zero = np.zeros((104918 - len(zzz))).astype("uint8")
        zzz = np.concatenate((zzz, zero))
    xxxx = zzz.reshape(-1, 418)

    cv2.namedWindow("1", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("1", xxxx)
    # cv2.imwrite("./1.jpg", xxxx)

    # keral = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # xxxx = cv2.morphologyEx(xxxx, cv2.MORPH_CLOSE, keral)
    # cv2.imshow("2", xxxx)

    cv2.waitKey(0)

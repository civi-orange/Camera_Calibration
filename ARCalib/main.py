#!/usr/bin/python

from spaam import SPAAM
import numpy as np
from read_txt import read_txt
from scipy import linalg


def main():
    """
    the point_2d_*.txt represents the coordinates of points on the AR screen
    the point_3d_*.txt represents the coordinates of points on the infrared camera
    """
    pImage_0 = read_txt(r"../image_calib/txt/cross_point_xy.txt", col=2)
    pImage_1 = read_txt(r"../image_calib/txt/cross_point_xy.txt", col=2)
    pImage_2 = read_txt(r"../image_calib/txt/cross_point_xy.txt", col=2)
    pImage_3 = read_txt(r"../image_calib/txt/cross_point_xy.txt", col=2)
    pImage = np.concatenate((pImage_0, pImage_1, pImage_2, pImage_3), axis=0)

    from calib_info import K_red
    KKK = np.linalg.inv(K_red)
    pWorld_0 = read_txt(r"../image_calib/txt/point_3d_circle.txt", col=3)
    pwdxy0 = pWorld_0.T[0:2, :]
    pwdxy1 = np.concatenate((pwdxy0, np.ones((1, pwdxy0.shape[1]))), axis=0)
    pWorld_00 = KKK.dot(pwdxy1) * pWorld_0[0, 2]

    pWorld_1 = read_txt(r"../image_calib/txt1/point_3d_circle.txt", col=3)
    pwdxy1 = pWorld_1.T[0:2, :]
    pwdxy1 = np.concatenate((pwdxy1, np.ones((1, pwdxy1.shape[1]))), axis=0)
    pWorld_11 = KKK.dot(pwdxy1) * pWorld_1[0, 2]

    pWorld_2 = read_txt(r"../image_calib/txt2/point_3d_circle.txt", col=3)
    pwdxy2 = pWorld_2.T[0:2, :]
    pwdxy1 = np.concatenate((pwdxy2, np.ones((1, pwdxy2.shape[1]))), axis=0)
    pWorld_22 = KKK.dot(pwdxy1) * pWorld_2[0, 2]

    pWorld_3 = read_txt(r"../image_calib/txt2/point_3d_circle.txt", col=3)
    pwdxy3 = pWorld_3.T[0:2, :]
    pwdxy1 = np.concatenate((pwdxy3, np.ones((1, pwdxy3.shape[1]))), axis=0)
    pWorld_33 = KKK.dot(pwdxy1) * pWorld_3[0, 2]

    pWorld = np.concatenate((pWorld_00.T, pWorld_11.T, pWorld_22.T, pWorld_33.T), axis=0)

    spaam = SPAAM(pImage, pWorld)
    G, ggggggg = spaam.get_camera_matrix()
    K, A = spaam.get_transformation_matrix()

    print("-------------------")
    print("G Matrix:")
    print(G)
    print("\n")
    print("Projection (Camera) Matrix (K):")
    print(K)
    print("\n")
    print("Transformation Matrix (R|t)")
    print(A)


if __name__ == '__main__':
    main()

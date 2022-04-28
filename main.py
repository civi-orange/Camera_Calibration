#!/usr/bin/python

import argparse
import csv
from spaam import SPAAM
from img2img_AR_calib import SCM


def main():
    # parser = argparse.ArgumentParser(
    #     description='Provide file path to csv file containing Pixel and corresponding 3D Coordinate Values')
    # parser.add_argument("filePath", type=str, help='path to relevant csv file')
    # args = parser.parse_args()

    # with open(args.filePath, newline='') as csvfile:
    #     rows = csv.DictReader(csvfile)
    #     pImage = []
    #     pWorld = []
    #     try:
    #         for row in rows:
    #             pImage.append([float(row['\ufeffDisplay Horizontal Pixel']),
    #                            float(row['Display Vertical Pixel'])])
    #             pWorld.append([float(row['HMD Position X']),
    #                            float(row['HMD Position Y']),
    #                            float(row['HMD Position Z'])])
    #
    #     except csv.Error as e:
    #         print(e)

    import numpy as np

    pImage = np.random.randint(1, 100, (6, 2))
    pWorld = np.random.randint(1, 100, (6, 3))

    spaam = SPAAM(pImage, pWorld)
    G,ggggggg = spaam.get_camera_matrix()
    K, A = spaam.get_transformation_matrix()

    print("G Matrix:")
    print(G)
    print("\n")
    print("Projection (Camera) Matrix (K):")
    print(K)
    print("\n")
    print("Transformation Matrix (R|t)")
    print(A)

    ppppp =[[pWorld[0,0], pWorld[0,1],pWorld[0,2], 1],]
    ppppp = np.array(ppppp)
    ppppp = ppppp.transpose()
    pp = G.dot(ppppp)
    pp = pp / pp[-1]

    pp1 = ggggggg.dot(ppppp)
    pp1 = pp1 / pp1[-1]

    print("-------****************----------------")
    print(pWorld[0])
    print(ppppp.transpose())
    print(pImage[0])
    print(pp[:2,:].transpose())
    print(pp1[:2, :].transpose())
    print("-------****************----------------")
    p1 = np.random.randint(1, 100, (4, 2))
    p2 = np.random.randint(1, 100, (4, 2))

    scm = SCM(p1, p2)
    A_ = scm.get_matrix_of_p2p()
    print("\n", "A_ Matrix:", "\n", A_, "\n")

    qqqqq = [[p1[0, 0]], [p1[0, 1]], [1],]
    qqqqq = np.array(qqqqq)
    qq = A_.dot(qqqqq)
    qq = qq / qq[-1]
    print()
    print(p1[0])
    print(qqqqq.transpose())
    print(p2[0])
    print(qq[:2,].transpose())

if __name__ == '__main__':
    main()

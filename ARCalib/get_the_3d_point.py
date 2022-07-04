# -*- coding: utf-8 -*-
# 从图像中获取坐标，
import os
from glob import glob
import cv2


point = []


def mouse_left(event, x, y, flags, param):
    frame = param
    depth = 2500
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d %d %d" % (x, y, depth)
        point.append(xy)
        cv2.putText(frame, xy, (x + 2, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 0, 0), thickness=1)
        cv2.imshow("image", frame)
        cv2.waitKey(1)


path = r"..\image_calib\near"
if __name__ == '__main__':
    files = os.listdir(path)
    ffiles = []
    for file in files:
        name, ext = os.path.splitext(file)
        name = int(name)
        ffiles.append(name)
    ffiles.sort()

    for i in ffiles:
        file = os.path.join(path, str(i) + '.jpg')
        image = cv2.imread(file)
        cv2.namedWindow("image")
        cv2.imshow("image", image)
        cv2.setMouseCallback("image", mouse_left, image)
        cv2.waitKey(0)

    txt = open(r"..\image_calib\txt\point_3d_near.txt", "w")
    for line in point:
        txt.write(line + '\n')
    txt.close()
    cv2.destroyAllWindows()

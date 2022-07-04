# -*- python coding: utf-8 -*-
# @Time: 6月 06, 2022
# ---
import time

import cv2
import numpy as np

from inv_Image_Mapping import *
from ImageMapping import *
from SDKCameraRead import *
# camera image
from fusion_function import *
import imageio

# 获取相机的内外参信息
from calib_info import left_Kl
from calib_info import left_Kr
from calib_info import left_R
from calib_info import left_T

depth = [5700]
running = False

if __name__ == "__main__":
    save_text = False
    save_video = False

    # capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # capture1 = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    camera_class = EnumMVCamera()
    cam1 = camera_class.init_camera(camera_index=1)
    cam2 = camera_class.init_camera(camera_index=0)

    num = 0
    now = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
    if save_text:
        file_error_depend_on_depth = "../error_txt/" + now + r"_depth.txt"
        f_err_depth = open(file_error_depend_on_depth, "w")
    if save_video:
        io_writer = imageio.get_writer('../video/{}.mp4'.format(str(now)), fps=24)

    test_map_left = ImageMapping(left_Kl, left_Kr, left_R, left_T, LEFT_2_RIGHT)
    new_map = ImageFusionInv(left_Kl, left_Kr, left_R, left_T, dst_size=(1024, 1280))
    while not running:
        t1 = MyThread(cam1)
        t1.start()
        t1.join()
        t2 = MyThread(cam2)
        t2.start()
        t2.join()
        # _, frame_l = capture.read()
        # _, frame_r = capture1.read()
        # frame_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
        # frame_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
        frame_1 = t1.get_result()
        frame_2 = t2.get_result()

        num += 1
        print("The frame number is:", num)

        save_calib_image = False
        if not save_calib_image:

            region_mask = np.zeros_like(frame_1)
            region_mask[740:830, 480:580] = 255
            out = new_map.image_fusion(frame_1, region_mask, frame_2, 3100)
            cv2.imshow("out", out)

            cv2.imshow("frame_1", frame_1)  # 模拟AR眼镜
            cv2.imshow("frame_2", frame_2)  # 模拟AR眼镜
            # cv2.imshow("frame_l", frame_l)  # 模拟微光相机
            # cv2.imshow("frame_r", frame_r)  # 模拟微光相机
            cv2.waitKey(1)
            # # 图像融合
            # print("----------------------------------------")
            # region_left, depth_left, rect = get_region_depth(frame_l, frame_1, test_map_left)
            # print(rect)
            # print("left_depth:", depth_left[0])
            # region1 = np.full_like(frame_l, 255)
            # out_left = test_map_left.fusion(frame_l, frame_1, region_left, depth_left[0])
            # cv2.putText(out_left, str(depth_left[0]), (200, 200), 1, 3, (255, 0, 0))
            # cv2.imshow("out_left", out_left)
            # cv2.waitKey(1)

        else:
            img_path = "../calib2/"
            cv2.imshow("f1", frame_1)
            cv2.imshow("f2", frame_2)

            # cv2.imwrite(img_path + "1/f_{:0>6}.bmp".format(num), frame_1)
            # cv2.imwrite(img_path + "2/rgb_{:0>6}.bmp".format(num), frame_2)
            # cv2.waitKey(2000)
            cv2.waitKey(1)
        # if save_video:
        #     io_writer.append_data(out)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):  # press 'q' to exit
            running = True
if save_text:
    f_err_depth.close()
if save_video:
    io_writer.close()
cv2.destroyAllWindows()

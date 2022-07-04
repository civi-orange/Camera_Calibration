# -*- python coding: utf-8 -*-
# @Time: 5月 18, 2022
# ---
import cv2
from tqdm import tqdm
import os
import numpy as np


# picture to video class
class Picture2Video:
    def __init__(self, root_path='.', realtime=False, video_name=None, fps=24, size=None):
        if realtime:
            assert size is not None, "size is None"
            h, w = size
        else:
            self.paths = self.__load_picture(root_path)
            frame = cv2.imread(self.paths[0], cv2.IMREAD_GRAYSCALE)
            h, w = frame.shape  # 需要转为视频的图片的尺寸

        self.save_path = root_path + "/VideoTest.mp4" if video_name is None else os.path.join(root_path, video_name)
        if not self.save_path.endswith('mp4'):
            self.save_path += '.mp4'
        self.video = cv2.VideoWriter(self.save_path, cv2.VideoWriter_fourcc(*'AVC1'), fps, (w, h), True)

    def __load_picture(self, root_path):
        paths = [d for d in os.listdir(root_path) if d.endswith(('jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff'))]
        paths.sort(key=lambda x: int(x.split('.')[0]), reverse=False)
        paths = [os.path.join(root_path, p) for p in paths]
        return paths

    def toVideo(self, start_frame=0, end_frame=-1):
        end_frame = len(self.paths) if end_frame == -1 else min(end_frame, len(self.paths))
        assert end_frame >= start_frame, "保存帧数不对"
        for i in tqdm(range(start_frame, end_frame)):
            img = cv2.imread(self.paths[i], cv2.IMREAD_GRAYSCALE)
            self.video.write(img)

        print('保存完毕!')
        print('保存地址为：', self.save_path)

    def pic2video(self, img):
        self.video.write(img)

    def __del__(self):
        self.video.release()
        cv2.destroyAllWindows()


def maxblur(frame_in: np.array, flag: int = 0):
    """
    本函数只适用于
    :param flag:0为非正常图像滤波，1为正常图像滤波
    :param frame_in: 图像输入
    :return: 返回图像
    """
    frame = frame_in.copy()
    if flag == 0:
        for j in range(1, frame.shape[1] - 1):
            for i in range(1, frame.shape[0] - 1):
                if frame[i][j] == 0:
                    # x = np.array(
                    #     [frame[i - 1][j - 1], frame[i - 1][j], frame[i - 1][j + 1], frame[i][j - 1], frame[i][j],
                    #      frame[i][j + 1], frame[i + 1][j - 1], frame[i + 1][j], frame[i + 1][j + 1]])
                    x = np.array([frame[i][j - 1], frame[i][j], frame[i][j + 1]])
                    frame[i][j] = np.max(x)
                    print("x=", x, frame[i][j])
    elif flag == 1:
        for j in range(1, frame.shape[1] - 1):
            for i in range(1, frame.shape[0] - 1):
                if frame[i][j] > 150:
                    x = np.array([frame[i][j - 1], frame[i][j], frame[i][j + 1]])
                    frame[i][j] = frame[i][j - 1] = frame[i][j - 1] = 255

    return frame.astype("uint8")


# round the float to int and always round(0.5)=1
def myround(matrix: np.array):
    """
    :param matrix: array
    :return: out: array
    """
    out = matrix.copy()
    int_out = out.astype(np.int32)
    ban = out - int_out
    mask = np.where(ban >= 0.5)
    int_out[mask] = int_out[mask] + 1
    return int_out


# findcircle
def gen_circle_center(frame_in, min_R=5, max_R=80, win_name="circle", b_display=False):
    frame = frame_in.copy()
    _, frame = cv2.threshold(frame, 110, 255, cv2.THRESH_BINARY)
    frame_c = cv2.Canny(frame, 30, 100)
    circles = cv2.HoughCircles(frame_c, cv2.HOUGH_GRADIENT, 1, 10000,
                               param1=100, param2=10, minRadius=min_R, maxRadius=max_R)
    try:
        circles = np.uint16(np.around(circles))
    except:
        print("未找到圆!")
        return [[0, 0]], [0]

    cimg = frame.copy()
    center = []
    radius = []
    if len(circles) != 0:
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(cimg, (i[0], i[1]), i[2], (255, 255, 255), 2)
            # draw the center of the circle
            cv2.circle(cimg, (i[0], i[1]), 2, (255, 255, 255), 3)
            center.append(i[0])
            center.append(i[1])
            radius.append(i[2])
        center = np.array(center).reshape(-1, 2)
        # print("图中圆的个数为：", len(circles))
        if b_display is True:
            cv2.imshow(win_name, cimg)
            cv2.waitKey(1)
        return center, radius


def gray2rgb_greenFrame(frame_in: np.array, color="g"):
    """
    :param frame_in: 灰度输入图像
    :param color: 可以去 r g b分别代表红绿蓝
    :return:
    """
    frame = frame_in.copy()
    frame_out = np.dstack(
        (np.zeros((frame.shape[0], frame.shape[1])), frame, np.zeros((frame.shape[0], frame.shape[1]))))

    return frame_out.astype("uint8")


def line_frame(frame: np.array, line_h_number=3, line_w_number=3, color=(255, 0, 0)):
    """
    给图像frame宫格分块
    :param frame: 图像输入
    :param line_h_number: h方向均分线条数
    :param line_w_number: w方向均分线条数
    :return: 画好宫格的图像
    """
    frame_copy = frame.copy()
    step_h = int(frame_copy.shape[0] / line_h_number)
    step_w = int(frame_copy.shape[1] / line_w_number)
    for i in range(step_h, frame_copy.shape[0], step_h):
        frame_copy = cv2.line(frame_copy, (0, i), (frame_copy.shape[1], i), color, 1)
    for i in range(step_w, frame_copy.shape[1], step_w):
        frame_copy = cv2.line(frame_copy, (i, 0), (i, frame_copy.shape[0]), color, 1)

    return frame_copy


def cross_frame(frame: np.array, line_h_number=3, line_w_number=3, color=(255, 0, 0)):
    """
    给图像frame画十字点
    :param frame:
    :param line_h_number:
    :param line_w_number:
    :param color:
    :return:
    """
    step_h = int(frame.shape[0] / line_h_number)
    step_w = int(frame.shape[1] / line_w_number)
    for j in range(step_h, frame.shape[0], step_h):
        for i in range(step_w, frame.shape[1], step_w):
            frame = cv2.line(frame, (i - 30, j), (i + 30, j), color, 2)
            frame = cv2.line(frame, (i, j - 30), (i, j + 30), color, 2)
            frame = cv2.circle(frame, (i, j), 4, color, 5)
            print(i, j)
    return frame


def save(image, num, file_path=None):
    if file_path is None:
        os.mkdir("./images")
        file_path = os.path.join("./images", r"{:>6}.bmp".format(num))
    else:
        file_path = os.path.join(file_path, r"{:>6}.bmp".format(num))
    cv2.imwrite(file_path, image)

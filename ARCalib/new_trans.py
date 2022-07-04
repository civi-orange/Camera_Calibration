import cv2
import numpy as np

from little_function import *
from calib_info import K_red
from calib_info import G
from AR_display import ZDmethod


def main():
    dst = np.full((1080, 1920), 255).astype("uint8")
    rect = [82, 153, 417, 251]  # [352, 153, 417, 251]  # (243, 406)
    depth = 4000
    x, y, w, h = rect
    h_dst, w_dst = dst.shape[:2]

    cap = cv2.VideoCapture(0)
    edge = ZDmethod()

    while True:
        src = cv2.imread(r"C:\Users\37236\Desktop\wangbo\img\20220614_15_11_09_476.jpg", cv2.IMREAD_GRAYSCALE)
        _, frame = cap.read()
        src = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        src = edge.pred(src)
        cv2.imshow("src", src)
        cv2.waitKey(1)
        new_src = src[y:y + h, x:x + w]
        cv2.imshow("new_src", new_src)


        cv2.waitKey(1)

        h_d, w_d = dst.shape[:2]
        h_s, w_s = src.shape[:2]
        wh_rate = w_d / h_d  # 保真映射长宽比
        new_h = h_s
        new_w = round(h_s * wh_rate)
        if new_w - w_s > 0:
            new_src = np.concatenate((np.zeros((new_h, new_w - w_s)).astype('uint8'), src), axis=1)

        rate_red = [h / h_dst, w / w_dst]  # (h_rate, w_rate) ---scale
        trans = [new_h - h_s, new_w - w_s]  # 偏移量

        new_K_red = K_red.copy()
        new_K_red[0][2] += trans[1]
        new_K_red[0][1] += trans[0]
        _new_K_red = np.linalg.inv(new_K_red)

        region = np.full_like(new_src, 255)
        print(new_src.shape)
        rrr_yx = np.argwhere(region)
        rrr_xy = rrr_yx[:, [1, 0]]
        rrr_xy1 = np.concatenate((rrr_xy, np.ones((rrr_xy.shape[0], 1))), axis=1)
        rrr_xy1 = np.transpose(rrr_xy1, (1, 0))
        rrr_xyd = _new_K_red.dot(rrr_xy1) * depth
        rrr_xyd1 = np.concatenate((rrr_xyd, np.ones((1, rrr_xyd.shape[1]))), axis=0)

        rate = [w / 1920, h / 1080]
        G1 = G.copy()
        G1[0:1, :] = G1[0:1, :] * rate[0]
        G1[1:2, :] = G1[1:2, :] * rate[1]
        rrr_xyz = G1.dot(rrr_xyd1)
        rrr_xyz = np.transpose(rrr_xyz, (1, 0))
        result_xyz = rrr_xyz / rrr_xyz[:, 2:]
        out = np.round(result_xyz).astype(np.int32)
        mask = np.array(out[:, 0] > 0) * np.array(out[:, 0] < w) * \
               np.array(out[:, 1] > 0) * np.array(out[:, 1] < h)

        # mask_arr = mask.reshape(h, -1)
        # mask_binary = np.full_like(new_src, 0)
        # mask_binary[mask_arr] = 255
        # num = np.sum(mask_arr == True)
        # print(num)
        # cv2.imshow("mask_binary", mask_binary)

        result = out[mask]
        srccc = rrr_xy[mask]
        imgggg = np.full_like(new_src, 0)
        imgggg[(result[:, 1], result[:, 0])] = new_src[(srccc[:, 1], srccc[:, 0])]
        cv2.imshow("222222222", new_src)
        imgggg = imgggg[:h, :w]
        cv2.imshow("t1", imgggg)  # 马赛克需要插值

        keral = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        imgggg1 = cv2.morphologyEx(imgggg, cv2.MORPH_CLOSE, keral)
        imgggg1 = np.concatenate((np.zeros_like(imgggg1[..., None]), imgggg1[..., None], np.zeros_like(imgggg1[..., None])), axis=2)
        cv2.imshow("t2", imgggg1)  # 马赛克需要插值

        imgggg2 = cv2.medianBlur(imgggg, 5)
        cv2.imshow("t3", imgggg2)  # 马赛克需要插值
        imgggg3 = cv2.blur(imgggg, (3, 3))
        cv2.imshow("t4", imgggg3)  # 马赛克需要插值
        ok_image = cv2.resize(imgggg1, (1920, 1080), interpolation=cv2.INTER_LINEAR)
        cv2.namedWindow('okkk', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("okkk", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.moveWindow("okkk", 2560, 0)
        cv2.imshow("okkk", ok_image)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()

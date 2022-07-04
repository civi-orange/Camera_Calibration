import cv2
import numpy as np
from calib_info import K_red
from calib_info import G
from little_function import *
import matplotlib.pyplot as plt


class ZDmethod:
    def __init__(self, video: bool = True):
        # if video:
        #     data_loader = self.readVideo
        # else:
        #     data_loader = self.readJpeg
        # self.data_loader = data_loader

        kirsch_0 = np.array([[5, 5, 5],
                             [-3, 0, -3],
                             [-3, -3, -3]], dtype='int')
        kirsch_1 = np.array([[-3, 5, 5],
                             [-3, 0, 5],
                             [-3, -3, -3]], dtype='int')
        kirsch_2 = np.array([[-3, -3, 5],
                             [-3, 0, 5],
                             [-3, -3, 5]], dtype='int')
        kirsch_3 = np.array([[-3, -3, -3],
                             [-3, 0, 5],
                             [-3, 5, 5]], dtype='int')
        kirsch_4 = np.array([[-3, -3, -3],
                             [-3, 0, -3],
                             [5, 5, 5]], dtype='int')
        kirsch_5 = np.array([[-3, -3, -3],
                             [5, 0, -3],
                             [5, 5, -3]], dtype='int')
        kirsch_6 = np.array([[5, -3, -3],
                             [5, 0, -3],
                             [5, -3, -3]], dtype='int')
        kirsch_7 = np.array([[5, 5, -3],
                             [5, 0, -3],
                             [-3, -3, -3]], dtype='int')
        self.operator = [kirsch_0, kirsch_1, kirsch_2, kirsch_3, kirsch_4, kirsch_5, kirsch_6, kirsch_7]

        self.clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(8, 8))

    def pred(self, frame):

        image_final = self.detect(frame)

        return image_final

    # def readVideo(self, root_path: str):
    #     # 1.初始化读取视频对象
    #     cap = cv2.VideoCapture(0)
    #
    #     # 2.循环读取图片
    #     while cap.isOpened():
    #         ret, frame = cap.read()
    #
    #         if ret:
    #             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 灰度处理
    #             yield frame
    #
    #         else:
    #             print("视频播放完成！")
    #             break
    #
    #         # 退出播放
    #         key = cv2.waitKey(1)
    #         if key == 27:  # 按键esc
    #             break

    # def readJpeg(self, root_path: str):
    #     paths = [d for d in os.listdir(root_path) if d.endswith(('jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff'))]
    #     # paths.sort(key=lambda x: int(x.split('.')[0]), reverse=False)
    #     # paths.sort(key=lambda x: int(x.split('.')[0].split('_')[1]), reverse=False)
    #     paths = [os.path.join(root_path, p) for p in paths]
    #
    #     for p in paths:
    #         frame = cv2.imread(p, -1)
    #         yield frame
    #
    #         # 退出播放
    #         key = cv2.waitKey(1)
    #         if key == 27:  # 按键esc
    #             break
    #
    #     if key == 27:
    #         print('视频停止播放！')
    #     else:
    #         print("视频播放完成！")

    @staticmethod
    def guidedFilter(im: np.array, r: int = 3, eps: float = 1000):
        """引导滤波算法，以自身图像作为引导算法，提取出背景层"""
        mean_I = cv2.boxFilter(im, cv2.CV_64F, (r, r), borderType=cv2.BORDER_REPLICATE)
        mean_II = cv2.boxFilter(im * im, cv2.CV_64F, (r, r), borderType=cv2.BORDER_REPLICATE)
        var_I = mean_II - mean_I ** 2

        a = var_I / (var_I + eps)
        b = (1 - a) * mean_I

        mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))  # 平滑的目的是，每个像素位置会经过r*r个窗口的计算，所以需要经过一此平滑，取平均
        mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

        q = mean_a * im + mean_b
        return np.clip(q, 0, 65535).astype('uint16')

    def edge_detect(self, img):
        edge_set = []
        for i in range(len(self.operator)):
            edge_set.append(cv2.filter2D(img, cv2.CV_64F, self.operator[i], borderType=cv2.BORDER_REPLICATE))

        edge_set = np.stack(edge_set, axis=0)
        edge_set = abs(edge_set)

        mmax_threshold = 40
        edge_set[edge_set < mmax_threshold] = mmax_threshold / (
                1 + np.exp(15 - 30 * edge_set[edge_set < mmax_threshold] / mmax_threshold)) - mmax_threshold / (
                                                      1 + np.exp(15))
        for i in range(len(self.operator)):
            edge_set[i][edge_set[i] < 1.5 * np.median(edge_set[i][edge_set[i] > 0])] = 0

        edge = np.sum(edge_set, axis=0) / (0.27 * np.mean(edge_set) + 11.6)
        edge = cv2.convertScaleAbs(edge)

        return edge

    def detect(self, img):
        img = (img / 4).astype('uint16')
        base = self.guidedFilter(img, eps=200)
        edge = self.edge_detect(base)
        edge_enhanced = self.clahe.apply(edge)
        return edge_enhanced

    def __del__(self):
        pass


class ImageTrans:
    def __init__(self, K1, G12, depth, dst_size=(1080, 1920)):
        """
        :param K1:  相机内参
        :param G12: AR标定矩阵(相机（左）到AR（右）)
        :param dst_size: (h, w) 映射目标尺寸
        :param depth: 深度
        """
        self.K1 = K1
        self.G12 = G12
        self.depth = depth
        self.dst_size = dst_size

    def xy_trans(self, region, new_K_red, src_size, dst_size=(1080, 1920)):
        """
        :param region: trans region: array
        :param dst_size: (h, w)
        :param new_K_red:
        :param src_size
        :return:
        """
        _new_K_red = np.linalg.inv(new_K_red)
        h, w = src_size
        rrr_yx = np.argwhere(region)
        rrr_xy = rrr_yx[:, [1, 0]]
        rrr_xy1 = np.concatenate((rrr_xy, np.ones((rrr_xy.shape[0], 1))), axis=1)
        rrr_xy1 = np.transpose(rrr_xy1, (1, 0))
        rrr_xyd = _new_K_red.dot(rrr_xy1) * self.depth
        rrr_xyd1 = np.concatenate((rrr_xyd, np.ones((1, rrr_xyd.shape[1]))), axis=0)

        rate = [w / dst_size[1], h / dst_size[0]]
        G1 = self.G12.copy()
        G1[0:1, :] = G1[0:1, :] * rate[0]
        G1[1:2, :] = G1[1:2, :] * rate[1]
        rrr_xyz = G1.dot(rrr_xyd1)
        rrr_xyz = np.transpose(rrr_xyz, (1, 0))
        result_xyz = rrr_xyz / rrr_xyz[:, 2:]
        out = myround(result_xyz).astype(np.int32)
        mask = np.array(out[:, 0] >= 0) * np.array(out[:, 0] < w) * np.array(out[:, 1] >= 0) * np.array(out[:, 1] < h)

        result_yx = out[mask]
        src_yx = rrr_xy[mask]

        return result_yx, src_yx, mask

    def get_FOV_rect(self, src_size, dst_size):
        h_dst, w_dst = dst_size
        h_src, w_src = src_size
        region = np.full(src_size, 255)

        wh_rate = w_dst / h_dst  # 保真映射长宽比
        new_h = h_src
        new_w = round(h_src * wh_rate)
        if new_w - w_src > 0:
            new_region = np.concatenate((np.full((new_h, new_w - w_src), 255).astype('uint8'), region), axis=1)
        else:
            new_region = region[:, w_src - new_w:]

        trans = [new_h - h_src, new_w - w_src]  # 偏移量
        new_K_red = self.K1.copy()
        new_K_red[0][2] += trans[1]
        new_K_red[0][1] += trans[0]
        h, w = new_region.shape[:2]

        result_yx, src_yx, mask = self.xy_trans(new_region, new_K_red, (h, w), dst_size)

        mask_arr = mask.reshape(512, -1)
        mask_binary = np.full_like(new_region, 0).astype("uint8")
        mask_binary[mask_arr] = 255
        # cv2.imshow("mask_binary", mask_binary)
        # cv2.waitKey(1)
        contours, hierarchy = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        assert contours is not None, "There is no contours!"
        for cont in contours:
            # 外接矩形
            rect = cv2.boundingRect(cont)

        return rect

    @classmethod
    def get_G_FPGA(cls, G12: np.array, depth: float, rate: int = None):
        """
        :param G12: 原始G矩阵
        :param depth: 应用深度
        :param rate: 元素乘以2^rate
        :return: 返回FPGA需要的G
        """
        G_out = G12.copy()
        G_out = G_out[:, -1] / depth
        if rate is not None:
            G_out = G_out * (2 ** 20)

        return G_out

    def image_trans(self, frame: np.array, rect: tuple):
        """
        映射
        :param frame: input
        :param rect: x,y,w,h
        :return:  out image
        """
        h_dst, w_dst = self.dst_size
        h_src, w_src = frame.shape[:2]
        wh_rate = w_dst / h_dst  # 保真映射长宽比
        new_w = round(h_src * wh_rate)
        trans = new_w - w_src
        new_K_red = self.K1.copy()
        x, y, w, h = rect
        x -= trans
        region = np.full_like(frame, 0)
        region[y:y + h, x:x + w] = 255

        result_yx, src_yx, mask = self.xy_trans(region, new_K_red, (h, w))

        out = np.zeros_like(frame)

        out[(result_yx[:, 1], result_yx[:, 0])] = frame[(src_yx[:, 1], src_yx[:, 0])]
        out_image = out[:h, :w]
        # cv2.imshow("out_image", out_image)
        keral = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        out_image = cv2.morphologyEx(out_image, cv2.MORPH_CLOSE, keral)
        # out_image = maxblur(out_image, 1)  # 自定义滤波

        FPGA = False
        if FPGA is True:
            rate = [w / 1920, h / 1080]
            G_fpga = self.G12.copy()
            G_fpga[0:1, :] = G_fpga[0:1, :] * rate[0]
            G_fpga[1:2, :] = G_fpga[1:2, :] * rate[1]
            rrr_yx = np.argwhere(region)
            rrr_xy = rrr_yx[:, [1, 0]]
            rrr_xy1 = np.concatenate((rrr_xy, np.ones((rrr_xy.shape[0], 1))), axis=1)
            rrr_xy1 = np.transpose(rrr_xy1, (1, 0))
            # FPGA data
            fpga_xyd = np.linalg.inv(self.K1).dot(rrr_xy1)
            fpga_xyd1 = np.concatenate((fpga_xyd, np.ones((1, fpga_xyd.shape[1]))), axis=0)
            G2 = G_fpga.copy()
            G2[:, 3:] = G2[:, 3:] / 3000
            fpga_out_xyd1 = G2.dot(fpga_xyd1)
            # fpga_out_xyz = np.round(fpga_out_xyd1).astype(np.int32)
            fpga_out_xyz = myround(fpga_out_xyd1).astype(np.int32)
            fpga_mask = np.array(fpga_out_xyz[0, :] >= 0) * np.array(fpga_out_xyz[:, :] < w) * np.array(
                fpga_out_xyz[1, :] >= 0) * np.array(fpga_out_xyz[1, :] < h)
            fpga_out_xyz_t = np.transpose(fpga_out_xyz, (1, 0))
            fpga_out_xyz_valid = fpga_out_xyz_t[mask]
            zero_y = []
            one_y = []
            yyyyy = fpga_out_xyz_t[:, 1]
            for i in range(len(fpga_out_xyz_t[:, 1])):
                if fpga_out_xyz_t[:, 1][i] == 0:
                    zero_y.append(fpga_out_xyz_t[:, 0][i])
                if fpga_out_xyz_t[:, 1][i] == 1:
                    one_y.append(fpga_out_xyz_t[:, 0][i])
            fpga_image = np.zeros_like(frame)
            fpga_image[fpga_out_xyz_valid[:, 1], fpga_out_xyz_valid[:, 0]] = frame[(src_yx[:, 1], src_yx[:, 0])]
            fpga_image_out = fpga_image[:h, :w]
            # cv2.namedWindow("fpga_image_out", cv2.WINDOW_KEEPRATIO)
            cv2.imshow("fpga_image_out", fpga_image_out)
            cv2.waitKey(1)

            # # print(fpga_out_xyd1.T)
            # print("--------------------------------------------")
            # print()
            # print(G2)
            # print()
            # print(G2*(2**20))
            # print()
            # print(np.linalg.inv(self.K1))
            # print()
            # print(np.linalg.inv(self.K1)*(2**30))
            # print("--------------------------------------------")
            # G2_20 = G2*(2**20)
            # G2_20 = G2_20.astype(np.int32)
            # K_30 = np.linalg.inv(self.K1)*(2**30)
            # K_30 = K_30.astype(np.int32)
            # fff = open("./mmm.txt", "w")
            # for i in range(len(K_30)):
            #     fff.write(str(K_30[i])+"\n")
            # fff.close()
            # fff = open("Gggg.txt", "w")
            # for i in range(len(G2_20)):
            #     fff.write(str(G2_20[i]) + "\n")
            # fff.close()
            #
            # out_txt = open(r"C:\Users\37236\Desktop\wangbo\code\out.txt", "w")
            # fpga_xy = fpga_out_xyd1.T[:, :2]
            # fpga_xy = np.round(fpga_xy).astype(np.int32)
            # for i in range(len(fpga_xy)):
            #     out_txt.write(str(fpga_xy[i]) + '\n')
            # out_txt.close()

        return out_image

    def image_trans_inv(self, frame: np.array, rect: tuple):

        # solve  FPGA实现反映射方法
        new_G12 = self.G12.copy()
        new_K_red = self.K1.copy()
        rect_x, rect_y, rect_w, rect_h = rect
        h_dst, w_dst = self.dst_size  # AR image display size
        h_src, w_src = frame.shape[:2]
        wh_rate = w_dst / h_dst  # 保真映射长宽比
        new_w = round(h_src * wh_rate)
        trans = new_w - w_src
        rect_x -= trans
        region = np.full_like(frame, 0)
        region[rect_y:rect_y + rect_h, rect_x:rect_x + rect_w] = 255
        rate = [rect_w / w_dst, rect_h / h_dst]

        new_G12[0:1, :] = new_G12[0:1, :] * rate[0]
        new_G12[1:2, :] = new_G12[1:2, :] * rate[1]
        new_G12[:, 3:] = new_G12[:, 3:] / 3000
        rrr_yx = np.argwhere(region)
        rrr_xy = rrr_yx[:, [1, 0]]
        rrr_xy1 = np.concatenate((rrr_xy, np.ones((rrr_xy.shape[0], 1))), axis=1)
        rrr_xy1 = np.transpose(rrr_xy1, (1, 0))

        gggg = np.array([
            [new_G12[0, 0], new_G12[0, 1], new_G12[0, 2] + new_G12[0, 3]],
            [new_G12[1, 0], new_G12[1, 1], new_G12[1, 2] + new_G12[1, 3]],
            [0, 0, 1]
        ])
        _gggg = np.linalg.inv(gggg)
        newkg = new_K_red.dot(_gggg)
        ar_image = np.full((rect_h, rect_w), 255)
        ar_yx = np.argwhere(ar_image > 0)
        ar_xy = ar_yx[:, [1, 0]]
        ar_xy1 = np.concatenate((ar_xy, np.ones((ar_xy.shape[0], 1))), axis=1)
        out_xy1 = newkg.dot(ar_xy1.T)
        out_xy1 = myround(out_xy1).astype(np.int32)
        ar_image[ar_xy[:, 1], ar_xy[:, 0]] = frame[out_xy1[1, :], out_xy1[0, :]]
        ar_image = ar_image.astype("uint8")

        # file = open("./m_g_inv_map.txt", "w")
        # for i in range(len(self.K1)):
        #     file.write(str(self.K1[i])+"\n")
        # for i in range(len(_gggg)):
        #     file.write(str(_gggg[i])+"\n")
        # file.close()
        #
        # file = open("./inv_map_out", "w")
        # file_xy = out_xy1.T[:, 0:2]
        # for i in range(len(file_xy)):
        #     file.write(str(file_xy[i]) + '\n')
        # file.close()
        return ar_image


def main():
    XT = ImageTrans(K_red, G, depth=5000)
    cap = cv2.VideoCapture(0)
    # import imageio
    # video_writer = imageio.get_writer("../output/inv_out.mp4", fps=24)
    zd = ZDmethod()
    while True:
        _, frame = cap.read()
        src = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # src = cv2.imread(r"..\img\20220614_15_11_09_476.jpg", cv2.IMREAD_GRAYSCALE)
        cv2.imshow("src", src)

        rect = XT.get_FOV_rect((src.shape[0], src.shape[1]), (1080, 1920))
        print(rect)

        out = XT.image_trans(src, rect)
        out = zd.detect(out)
        out = gray2rgb_greenFrame(out)

        out = cv2.resize(out, (1920, 1080), interpolation=cv2.INTER_LINEAR)
        cv2.imshow("out", out)

        # inv_out = XT.image_trans_inv(src, rect)
        # inv_out = zd.detect(inv_out)
        # inv_out = gray2rgb_greenFrame(inv_out)
        # inv_out = cv2.resize(inv_out, (1920, 1080), interpolation=cv2.INTER_LINEAR)
        # cv2.imshow("inv_out", inv_out)

        # cv2.namedWindow('okkk', cv2.WND_PROP_FULLSCREEN)
        # cv2.setWindowProperty("okkk", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        # cv2.moveWindow("okkk", int(2560), 0)
        # cv2.imshow("okkk", out)

        k = cv2.waitKey(10)
        if k == 27:
            exit()


if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()

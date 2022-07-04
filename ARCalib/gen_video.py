import cv2
import imageio
from AR_Trans_display import *


def gen_video():
    cap = cv2.VideoCapture(0)
    io_w = imageio.get_writer("../output/src.mp4", fps=24)
    while True:
        _, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("out", frame)
        io_w.append_data(frame)
        k = cv2.waitKey(10)
        if k == 27:
            exit()


def main():
    XT = ImageTrans(K_red, G, depth=5000)
    cap = cv2.VideoCapture(0)
    cap.open("../output/src.mp4")
    io_writer = imageio.get_writer("../output/dst.mp4", fps=24)
    zd = ZDmethod()
    while cap.isOpened():
        _, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frame = zd.detect(frame)

        rect = XT.get_FOV_rect((frame.shape[0], frame.shape[1]))

        out = XT.image_trans(frame, rect)
        out = gray2rgb_greenFrame(out)
        out = cv2.putText(out, "direct", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
        inv_out = XT.image_trans_inv(frame, rect)
        inv_out = gray2rgb_greenFrame(inv_out)
        inv_out = cv2.putText(inv_out, "inverse", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
        out = cv2.resize(out, (910, 540), interpolation=cv2.INTER_LINEAR)
        inv_out = cv2.resize(inv_out, (910, 540), interpolation=cv2.INTER_LINEAR)
        frame_out = np.concatenate((out, inv_out), axis=0)

        cv2.imshow("1", frame_out)

        io_writer.append_data(frame_out)
        k = cv2.waitKey(10)
        if k == 27:
            exit()



if __name__ == '__main__':
    main()

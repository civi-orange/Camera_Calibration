import cv2
import numpy as np

frame = cv2.imread(r"..\img\20220614_15_11_09_476.jpg", 0)
out = []
print(frame)
for i in range(frame.shape[0]):
    for j in range(frame.shape[1]):
        data_f = hex(frame[i][j])
        data = data_f.strip('0x')
        out.append(data)

f = open("hex.txt", 'w')
for i in range(len(out)):
    f.write(out[i]+'\n')
f.close()


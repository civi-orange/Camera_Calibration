# -*- python coding: utf-8 -*-
# @Time: 6月 06, 2022
# ---
import threading

import numpy as np
import sys

sys.path.append("./MvImport")
from MvImport.MvCameraControl_class import *


class EnumMVCamera:
    def __init__(self):
        self.device_list = MV_CC_DEVICE_INFO_LIST()
        self.device_type = MV_GIGE_DEVICE | MV_USB_DEVICE
        ret = MvCamera.MV_CC_EnumDevices(self.device_type, self.device_list)
        if ret != 0:
            print("enum devices fail! ret[0x%x]" % ret)
            sys.exit()
        if self.device_list.nDeviceNum == 0:
            print("find no device!")
            sys.exit()
        print("Find %d devices!" % self.device_list.nDeviceNum)

        if self.device_list.nDeviceNum > 0:
            self.enum_camera()

    def enum_camera(self):
        for i in range(0, self.device_list.nDeviceNum):
            mvcc_dev_info = cast(self.device_list.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
                print("\ngige device: [%d]" % i)
                strModeName = ""
                for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName:
                    strModeName = strModeName + chr(per)
                print("device model name: %s" % strModeName)
                nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
                nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
                nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
                nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
                print("current ip: %d.%d.%d.%d\n" % (nip1, nip2, nip3, nip4))
            elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
                print("\nu3v device: [%d]" % i)
                strModeName = ""
                for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName:
                    if per == 0:
                        break
                    strModeName = strModeName + chr(per)
                print("device model name: %s" % strModeName)
                strSerialNumber = ""
                for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
                    if per == 0:
                        break
                    strSerialNumber = strSerialNumber + chr(per)
                print("user serial number: %s" % strSerialNumber)

    def init_camera(self, camera_index=0, exposure_time=10000.0):
        cam = MvCamera()
        st_device_list = cast(self.device_list.pDeviceInfo[camera_index], POINTER(MV_CC_DEVICE_INFO)).contents
        cam.MV_CC_CreateHandle(st_device_list)
        cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        st_bool = c_bool(False)
        cam.MV_CC_GetBoolValue("AcquisitionFrameRateEnable", st_bool)
        cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        cam.MV_CC_SetFloatValue("ExposureTime", exposure_time)
        cam.MV_CC_StartGrabbing()
        return cam

    @classmethod
    def read_image(cls, cam=0):
        stOutFrame = MV_FRAME_OUT()
        stParam = MVCC_INTVALUE()
        memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))
        cam.MV_CC_GetIntValue("PayloadSize", stParam)
        ret = cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
        image = None
        if stOutFrame.pBufAddr is not None and 0 == ret:
            # print("get one frame: Width[%d], Height[%d], nFrameNum[%d]" % (
            #     stOutFrame.stFrameInfo.nWidth, stOutFrame.stFrameInfo.nHeight, stOutFrame.stFrameInfo.nFrameNum))
            nRet = cam.MV_CC_FreeImageBuffer(stOutFrame)
            pData = (c_ubyte * stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight)()
            cdll.msvcrt.memcpy(byref(pData), stOutFrame.pBufAddr,
                               stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight)
            data = np.frombuffer(pData, count=int(stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight),
                                 dtype=np.uint8)
            image = data.reshape((stOutFrame.stFrameInfo.nHeight, stOutFrame.stFrameInfo.nWidth))
        assert image is not None, "the image is None!"
        return image


# 线程获取结果
class MyThread(threading.Thread):
    def __init__(self, number):
        threading.Thread.__init__(self)
        self.number = number

    def run(self):
        self.result = EnumMVCamera.read_image(self.number)

    def get_result(self):
        return self.result
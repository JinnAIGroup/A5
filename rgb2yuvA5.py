'''  JHT, JLL, 2022.1.23 - 2.19
from /home/jinn/YPN/TangJH/combineA4h5/hevc2yuvh5A4.py

Input: bRGB images *.png
  /home/jinn/dataAll/comma10k/Ximgs/*.png
    bRGB (874, 1164, 3) = (H, W, C) <=> bYUV (1311, 1164) <=>  CbYUV (6, 291, 582) = (C, H, W) [key: 1311 =  874x3/2]
    sRGB (256,  512, 3) = (H, W, C) <=> sYUV  (384,  512) <=>  CsYUV (6, 128, 256) = (C, H, W) [key:  384 =  256x3/2]
Output: CsYUV
'''
import os
import cv2
import h5py
import numpy as np
from matplotlib import pyplot as plt
from cameraB5 import transform_img, eon_intrinsics, medmodel_intrinsics

def RGB_to_sYUV(img):
    bYUV = cv2.cvtColor(img, cv2.COLOR_RGB2YUV_I420)
      #---  np.shape(bYUV) = (1311, 1164)
      #print("#---  bYUV =", bYUV)   # check YUV: [0, 255]? Yes.
    sYUV = np.zeros((384, 512), dtype=np.uint8) # np.uint8 = 0~255
    sYUV = transform_img(bYUV, from_intr=eon_intrinsics, to_intr=medmodel_intrinsics,
                         yuv=True, output_size=(512, 256))  # (W, H)
    plt.clf()
    plt.subplot(131)
    plt.title("RGB Image 874x1164")
    plt.imshow(img)
    plt.subplot(132)
    plt.title("Big YUV 1311x1164")
    plt.imshow(bYUV)
    plt.subplot(133)
    plt.title("Small YUV 384x512")
    plt.imshow(sYUV)
    plt.show()
    return sYUV

def sYUV_to_CsYUV(sYUV):
    H = (sYUV.shape[0]*2)//3  # 384x2//3 = 256
    W = sYUV.shape[1]         # 512
    CsYUV = np.zeros((6, H//2, W//2), dtype=np.uint8)

    CsYUV[0] = sYUV[0:H:2, 0::2]  # [2::2] get every even starting at 2
    CsYUV[1] = sYUV[1:H:2, 0::2]  # [start:end:step], [2:4:2] get every even starting at 2 and ending at 4
    CsYUV[2] = sYUV[0:H:2, 1::2]  # [1::2] get every odd index, [::2] get every even
    CsYUV[3] = sYUV[1:H:2, 1::2]  # [::n] get every n-th item in the entire sequence
    CsYUV[4] = sYUV[H:H+H//4].reshape((-1, H//2, W//2))
    CsYUV[5] = sYUV[H+H//4:H+H//2].reshape((-1, H//2, W//2))

    CsYUV = np.array(CsYUV).astype(np.float32)   # RGB: [0, 255], YUV: [0, 255] => float32 (see __kernel void loadys())
      #visualize(CsYUV)   # TypeError: Invalid shape (6, 128, 256). We cannot visulize it.
    return CsYUV

def rgb2yuv(img):
      #---  np.shape(img) = (874, 1164, 3)
    sYUV = RGB_to_sYUV(img)
      #---  np.shape(sYUV) = (384, 512)
    CsYUV = sYUV_to_CsYUV(sYUV)
      #---  np.shape(CsYUV) = (6, 128, 256)
    return CsYUV

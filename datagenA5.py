"""   YPL, JLL, 2021.9.15 - 2022.2.19
for modelA5 = modelAB = UNet + RNN + PoseNet
from /home/jinn/YPN/ABNet/datagenAB2A.py

convert
  /home/jinn/dataAll/comma10k/Ximgs/*.png
directly to yuv tensors
  Ximgs.shape = (none, 2x6, 128, 256)
without saving to
  /home/jinn/dataAll/comma10k/Ximgs_yuv/*.h5

Input:
  /home/jinn/dataAll/comma10k/Ximgs/*.png  (X for debugging with 10 imgs)
  /home/jinn/dataAll/comma10k/Xmasks/*.png
    bRGB (874, 1164, 3) = (H, W, C) <=> bYUV (1311, 1164) <=>  CbYUV (6, 291,  582) = (C, H, W) [key: 1311 =  874x3/2]
    sRGB (256,  512, 3) = (H, W, C) <=> sYUV  (384,  512) <=>  CsYUV (6, 128,  256) = (C, H, W) [key:  384 =  256x3/2]
Output:
  Ximgs.shape = (none, 2x6, 128, 256)  (num_channels = 6, 2 yuv images)
  Xin1 = (none, 8)
  Xin2 = (none, 2)
  Xin3 = (none, 512)
  Ytrue0 = outs[ 0 ].shape = (none, 385)   # Ytrue0 - Ytrue11 are for Project B not A.
  Ytrue1 = outs[ 1 ].shape = (none, 386)
  Ytrue2 = outs[ 2 ].shape = (none, 386)
  Ytrue3 = outs[ 3 ].shape = (none, 58)
  Ytrue4 = outs[ 4 ].shape = (none, 200)
  Ytrue5 = outs[ 5 ].shape = (none, 200)
  Ytrue6 = outs[ 6 ].shape = (none, 200)
  Ytrue7 = outs[ 7 ].shape = (none, 8)
  Ytrue8 = outs[ 8 ].shape = (none, 4)
  Ytrue9 = outs[ 9 ].shape = (none, 32)
  Ytrue10 = outs[ 10 ].shape = (none, 12)
  Ytrue11 = outs[ 11 ].shape = (none, 512)
  Ymasks.shape = (None, 256, 512, 12)
"""
import os
import cv2
import h5py
import numpy as np
import matplotlib.pyplot as plt
from rgb2yuvA5 import rgb2yuv

def concatenate(img, msk, mask_H, mask_W, class_values):
    if os.path.isfile(img) and os.path.isfile(msk):
        img = cv2.imread(img)
        yuvCV2 = rgb2yuv(img)
          #---  yuvCV2.shape = (6, 128, 256)
        mskCV2 = cv2.imread(msk, 0).astype('uint8')
          #---1  mskCV2.shape = (874, 1164)
        mskCV2 = cv2.resize(mskCV2, (mask_W, mask_H))
          #---2  mskCV2.shape = (256, 512)
        mskCV2 = np.stack([(mskCV2 == v) for v in class_values], axis=-1).astype('uint8')
          #---3  mskCV2.shape = (256, 512, 6)
    else:
        print('#---datagenA5  Error: image.png or mask.png does not exist')

    return yuvCV2, mskCV2

def datagen(batch_size, images, masks, image_H, image_W, mask_H, mask_W, class_values):
    Ximgs  = np.zeros((batch_size, 12, image_H, image_W), dtype='float32')   # for YUV imgs
    Ymasks = np.zeros((batch_size, mask_H, mask_W, 2*len(class_values)), dtype='float32')
    Xin1   = np.zeros((batch_size, 8), dtype='float32')   # desire shape
    Xin2   = np.zeros((batch_size, 2), dtype='float32')   # traffic convection
    Xin3   = np.zeros((batch_size, 512), dtype='float32')   # rnn state
    Ytrue0 = np.zeros((batch_size, 385), dtype='float32')
    Ytrue1 = np.zeros((batch_size, 386), dtype='float32')
    Ytrue2 = np.zeros((batch_size, 386), dtype='float32')
    Ytrue3 = np.zeros((batch_size, 58), dtype='float32')
    Ytrue4 = np.zeros((batch_size, 200), dtype='float32')
    Ytrue5 = np.zeros((batch_size, 200), dtype='float32')
    Ytrue6 = np.zeros((batch_size, 200), dtype='float32')
    Ytrue7 = np.zeros((batch_size, 8), dtype='float32')
    Ytrue8 = np.zeros((batch_size, 4), dtype='float32')
    Ytrue9 = np.zeros((batch_size, 32), dtype='float32')
    Ytrue10 = np.zeros((batch_size, 12), dtype='float32')
    Ytrue11 = np.zeros((batch_size, 512), dtype='float32')

    Xin2[:, 0] = 1.0   # traffic convection = left hand drive like in Taiwan
    imgsN = len(images)
    print('#---datagenA5  imgsN =', imgsN)

    batchIndx = 0
    while True:
        count = 0
        while count < batch_size:
            print('#---  count =', count)
            ri = np.random.randint(0, imgsN-1, 1)[-1]   # ri cannot be the last img imgsN-1
            for i in range(imgsN-1):
                if ri < imgsN-1:   # the last imge is used only once
                    vsX1, vsY1 = concatenate(images[ri], masks[ri], mask_H, mask_W, class_values)
                      #---  vsX1.shape, vsY1.shape = (6, 128, 256) (256, 512, 6)
                    vsX2, vsY2 = concatenate(images[ri+1], masks[ri+1], mask_H, mask_W, class_values)

                    Ximgs[count] = np.vstack((vsX1, vsX2))
                    Ymasks[count] = np.concatenate((vsY1, vsY2), axis=-1)
                break
            count += 1

        yield Ximgs, Xin1, Xin2, Xin3, Ytrue0, Ytrue1, Ytrue2, Ytrue3, Ytrue4, Ytrue5, Ytrue6, Ytrue7, Ytrue8, Ytrue9, Ytrue10, Ytrue11, Ymasks

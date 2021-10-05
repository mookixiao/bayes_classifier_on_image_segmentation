import cv2
import numpy as np
from scipy import io
from scipy.stats import multivariate_normal


# 读取数据
sample_mat = io.loadmat('array_sample.mat')
sample_bgr_mat = sample_mat['array_sample'][:, 1:4]  # 7696 * 3
sample_label_mat = sample_mat['array_sample'][:, -1]  # 7696 * 1
sample_bgr = sample_bgr_mat.tolist()
sample_label = sample_label_mat.tolist()

white_pixes = []
red_pixes = []
for i in range(len(sample_bgr)):
    if sample_label[i] == -1:
        white_pixes.append(sample_bgr[i])
    else:
        red_pixes.append(sample_bgr[i])
white_pixes = np.array(white_pixes)  # m * 3，且m + n = 7696
red_pixes = np.array(red_pixes)  # n * 3

# 计算先验概率
white_prior = len(white_pixes) / (len(sample_bgr))
red_prior = len(red_pixes) / len(sample_bgr)

# 计算均值、方差
white_means = np.mean(white_pixes, axis=0)
red_means = np.mean(red_pixes, axis=0)
white_vars = np.var(white_pixes, axis=0)
red_vars = np.var(red_pixes, axis=0)

# 读入待分类图片
mask_mat = io.loadmat('Mask.mat')
mask_mat = mask_mat['Mask']  # 240 * 320
mask_mat = np.expand_dims(mask_mat, axis=2)  # 240 * 320 * 1
mask_mat = np.append(np.append(mask_mat, mask_mat, axis=2), mask_mat, axis=2)  # 240 * 320 * 3
img = cv2.imread('imgs/309.bmp')  # 210 * 320 * 3
img_masked = img * mask_mat / 255
img_processed = np.zeros([img.shape[0], img.shape[1]])

# 二分类
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if img_masked[i][j].sum() == 0:
            continue
        white_val = white_prior * multivariate_normal.pdf(img_masked[i][j], white_means, np.sqrt(white_vars))
        red_val = red_prior * multivariate_normal.pdf(img_masked[i][j], red_means, np.sqrt(red_vars))
        if white_val > red_val:
            img_processed[i][j] = 0  # 此处应为1和0，为明显起见，反转两者
        else:
            img_processed[i][j] = 1

cv2.imshow('img', img)
cv2.imshow('img masked', img_masked)
cv2.imshow('img processed', img_processed)

cv2.waitKey()
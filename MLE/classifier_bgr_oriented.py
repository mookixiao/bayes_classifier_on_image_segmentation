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
red_prior = 1 - white_prior

# 计算均值、方差
white_means = np.mean(white_pixes, axis=0)
red_means = np.mean(red_pixes, axis=0)
white_vars = np.cov(white_pixes, rowvar=False)
red_vars = np.cov(red_pixes, rowvar=False)

# 读入待分类图片
mask_mat = io.loadmat('Mask.mat')
mask_mat = mask_mat['Mask']  # 240 * 320
mask_mat = np.array([mask_mat, mask_mat, mask_mat]).transpose(1, 2, 0)  # 240 * 320 * 3
img_bgr = cv2.imread('imgs/309.bmp')  # 210 * 320 * 3
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # BGR -> RGB, 和给定sample的通道顺序保持一致
img_rgb_masked = img_rgb * mask_mat / 255
img_rgb_processed = np.zeros_like(img_rgb)


# 二分类
for i in range(img_bgr.shape[0]):
    for j in range(img_bgr.shape[1]):
        if img_rgb_masked[i][j].sum() == 0:
            continue
        white_val = white_prior * multivariate_normal.pdf(img_rgb_masked[i][j], white_means, white_vars)
        red_val = red_prior * multivariate_normal.pdf(img_rgb_masked[i][j], red_means, red_vars)
        if white_val > red_val:
            img_rgb_processed[i][j][0] = 255
            img_rgb_processed[i][j][1] = 255
            img_rgb_processed[i][j][2] = 255
        else:
            img_rgb_processed[i][j][0] = 255
            img_rgb_processed[i][j][1] = 0
            img_rgb_processed[i][j][2] = 0

img_bgr_masked = cv2.cvtColor(img_rgb_masked.astype(np.float32), cv2.COLOR_RGB2BGR)
img_bgr_processed = cv2.cvtColor(img_rgb_processed, cv2.COLOR_RGB2BGR)

cv2.imshow('img bgr', img_bgr)
cv2.imshow('img bgr masked', img_bgr_masked)
cv2.imshow('img bgr processed', img_bgr_processed)


cv2.waitKey()
import numpy as np
import os
from PIL import Image
from scipy import io
from scipy.stats import multivariate_normal
from EM.utils import save_rgb_img

# 输出文件夹
dir_out = 'EM_rgb_out'
if not os.path.isdir(dir_out):
    os.mkdir(dir_out)

# 超参
epochs = 30

# 训练数据
sample = io.loadmat('../array_sample.mat')['array_sample']
sample_rgb = sample[:, 1:4]  # 7696 * 3

# 待处理图片
img = Image.open('../imgs/309.bmp')
img = np.array(img)
mask = io.loadmat('../Mask.mat')['Mask']
mask = np.array([mask, mask, mask]).transpose(1, 2, 0)
img_masked = img * mask / 255

save_rgb_img(dir_out, 'img_masked.png', img_masked)

# 初始化先验概率
white_prior = 0.5
red_prior = 1 - white_prior

# 初始化均值、协方差矩阵
white_means = np.array([0.8, 0.8, 0.8])
white_cov = np.array([[0.1, 0.05, 0.04],
                      [0.05, 0.1, 0.02],
                      [0.04, 0.02, 0.1]])
red_means = np.array([0.5, 0.5, 0.5])
red_cov = np.array([[0.1, 0.05, 0.04],
                    [0.05, 0.1, 0.02],
                    [0.04, 0.02, 0.1]])

# 迭代开始
tmp_labels = np.zeros((sample_rgb.shape[0], 2))
tmp_white = np.zeros((sample_rgb.shape[0], 3))
tmp_red = np.zeros((sample_rgb.shape[0], 3))
for epoch in range(epochs):
    ### E过程
    for i in range(sample_rgb.shape[0]):
        white_score = white_prior * multivariate_normal.pdf(sample_rgb[i], white_means, white_cov)
        red_score = red_prior * multivariate_normal.pdf(sample_rgb[i], red_means, red_cov)

        white_label = white_score / (white_score + red_score)
        red_label = 1 - white_label

        tmp_labels[i, 0], tmp_labels[i, 1] = white_label, red_label  # 此数据的多少比例属于白色/红色
        tmp_white[i] = sample_rgb[i] * white_label  # 此数据的多少属于白色/红色
        tmp_red[i] = sample_rgb[i] * red_label

    ### M过程
    # 更新先验概率
    white_num = tmp_labels[:, 0].sum()
    red_num = tmp_labels[:, 1].sum()
    white_prior = white_num / (white_num + red_num)
    red_prior = 1 - white_prior

    # 更新均值
    white_means = tmp_white.sum(axis=0) / white_num
    red_means = tmp_red.sum(axis=0) / red_num

    # 更新协方差矩阵
    white_cov_sum = np.zeros([3, 3])
    red_cov_sum = np.zeros([3, 3])
    for i in range(len(sample_rgb)):
        white_cov_sum = white_cov_sum + np.dot((sample_rgb[i] - white_means).reshape(3, 1),
                                               (sample_rgb[i] - white_means).reshape(1, 3)) * tmp_labels[i, 0]
        red_cov_sum = red_cov_sum + np.dot((sample_rgb[i] - red_means).reshape(3, 1),
                                           (sample_rgb[i] - red_means).reshape(1, 3)) * tmp_labels[i, 1]

    white_cov = white_cov_sum / (white_num - 1)  # 无偏估计除以N-1
    red_cov = red_cov_sum / (red_num - 1)

    ### 使用当前参数处理图像
    img_out = np.zeros_like(img_masked)
    for m in range(img_masked.shape[0]):
        for n in range(img_masked.shape[1]):
            if img_masked[m][n].sum() == 0:
                continue
            white_val = white_prior * multivariate_normal.pdf(img_masked[m][n], white_means, white_cov)
            red_val = red_prior * multivariate_normal.pdf(img_masked[m][n], red_means, red_cov)
            if white_val > red_val:
                img_out[m][n] = [255, 255, 255]
            else:
                img_out[m][n] = [255, 0, 0]

    save_rgb_img(dir_out, 'img_processed_Epoch' + str(epoch + 1) + '.png', img_out)
# 基于311.bmp、313.bmp、315.bmp、317.bmp灰度图进行处理
import cv2
import numpy as np
import os
from PIL import Image
from scipy import io
from scipy.stats import multivariate_normal

from utils import *

# 读取数据
img = Image.open('imgs/309.bmp')
img = np.array(img)  # 240 * 320 * 3
img_mask = io.loadmat('Mask.mat')['Mask']  # 240 * 320

# 分类像素点
fish_pixels = []
bg_pixels = []
img_lst = img.tolist()
img_mask = img_mask.tolist()
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if img_mask[i][j] == 1:
            fish_pixels.append(img_lst[i][j])
        else:
            bg_pixels.append(img_lst[i][j])

fish_pixels = np.array(fish_pixels)
bg_pixels = np.array(bg_pixels)

# 计算先验概率
fish_prior = len(fish_pixels) / (len(fish_pixels) + len(bg_pixels))
bg_prior = 1 - fish_prior

# 计算均值、协方差矩阵
fish_means = np.mean(fish_pixels, axis=0)
fish_vars = np.cov(fish_pixels, rowvar=False)
bg_means = np.mean(bg_pixels, axis=0)
bg_vars = np.cov(bg_pixels, rowvar=False)





# weight = 1.2
# imgs_dir = 'imgs'
# masks_dir = 'Masks_generated'
# if __name__ == '__main__':
#     for img_name in ['311.bmp', '313.bmp', '315.bmp', '317.bmp']:
#         get_mask(weight, imgs_dir, img_name, masks_dir)

# 超参数
img_dir = 'imgs'
img_sample = '309.bmp'
imgs_to_process = ['311.bmp', '313.bmp', '315.bmp', '317.bmp']
height_range = [0, 170]
width_range = [50, 250]

sample_mat_file = 'array_sample.mat'
sample_mat_name = 'array_sample'

mask_file = 'Mask.mat'
mask_name = 'Mask'

weight_gray = 0.2
weight_rgb = 0.2
out_dir = 'out/311_313_315_317_NoMask'


if __name__ == '__main__':
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # 由样本计算假设分布模型的参数
    gray_arr, rgb_arr, label_arr = get_fish_sample(sample_mat_file, sample_mat_name)
    white_pixels_gray, red_pixels_gray = get_white_red_pixels(gray_arr, label_arr)
    white_prior, red_prior = get_prior(white_pixels_gray, red_pixels_gray)
    white_mean_gray, red_mean_gray, white_std_gray, red_std_gray = get_mean_std(white_pixels_gray, red_pixels_gray)

    # 由样本计算前景（Nemo鱼）和背景假设分布模型的参数
    fg_pixels, bg_pixels = get_whole_sample(img_dir + '/' + img_sample, mask_file, mask_name)
    fg_prior, bg_prior = get_prior(fg_pixels, bg_pixels)
    fg_mean_rgb, bg_mean_rgb, fg_cov_rgb, bg_cov_rgb = get_mean_cov(fg_pixels, bg_pixels)
    for img in imgs_to_process:
        generate_mask(out_dir, img_dir, imgs_to_process, height_range, width_range,
                       (fg_prior, bg_prior), (fg_mean_rgb, bg_mean_rgb), (fg_cov_rgb, bg_cov_rgb))


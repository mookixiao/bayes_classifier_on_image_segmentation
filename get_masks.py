import cv2
import numpy as np
import os
from PIL import Image
from scipy import io
from scipy.stats import multivariate_normal

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


def get_mask(weight, imgs_dir, img_name, masks_dir):
    print('the mask of ' +imgs_dir + '/' + img_name + ' generating...')
    if not os.path.isdir(masks_dir):
        os.mkdir(masks_dir)

    # 读入待分类图片
    img_raw = Image.open(imgs_dir + '/' + img_name)
    img_raw = np.array(img_raw)

    # 二分类
    img_raw_mask = np.zeros(img_raw.shape[:2])
    for i in range(img_raw.shape[0]):
        for j in range(img_raw.shape[1]):
            if i < 170 and j > 50 and j < 250:  # 只对特定区域进行处理
                fish_prob = fish_prior * multivariate_normal.pdf(img_raw[i][j], fish_means, fish_vars)
                bg_prob = bg_prior * multivariate_normal.pdf(img_raw[i][j], bg_means, bg_vars)
                if weight * fish_prob > bg_prob:
                    img_raw_mask[i][j] = 1

    kernel = np.ones([5, 5])
    img_raw_mask = cv2.dilate(img_raw_mask, kernel)
    img_raw_mask = cv2.erode(img_raw_mask, kernel)

    # cv2.imshow('after weight=' + str(weight), img_raw_mask)
    prefix = masks_dir + '/' + 'Mask_' + img_name.split('/')[-1].split('.')[-2]
    cv2.imwrite(prefix + '.png', img_raw_mask * 255)
    io.savemat(prefix + '.mat', {'Mask': img_raw_mask})


weight = 1.2
imgs_dir = 'imgs'
masks_dir = 'Masks_generated'
if __name__ == '__main__':
    for img_name in ['311.bmp', '313.bmp', '315.bmp', '317.bmp']:
        get_mask(weight, imgs_dir, img_name, masks_dir)
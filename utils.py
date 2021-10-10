import cv2
import numpy as np
from scipy import io
from scipy.stats import norm, multivariate_normal


# 读取样本数据
def get_sample(mat_file, mat_name):
    mat = io.loadmat(mat_file)

    gray_arr = mat[mat_name][:, 0]
    rgb_arr = mat[mat_name][:, 1:4]
    label_arr = mat[mat_name][:, -1]

    return gray_arr, rgb_arr, label_arr


# 分类像素点
def get_white_red_pixels(pixel_arr, label_arr):
    pixel_lst = pixel_arr.tolist()
    label_lst = label_arr.tolist()

    white_pixels = []
    red_pixels = []
    for i in range(len(pixel_lst)):
        if label_lst[i] == -1:
            white_pixels.append(pixel_lst[i])
        else:
            red_pixels.append(pixel_lst[i])
    white_pixels = np.array(white_pixels)
    red_pixels = np.array(red_pixels)

    return white_pixels, red_pixels


# 计算先验概率
def get_prior(white_pixels, red_pixels):
    white_prior = len(white_pixels) / (len(red_pixels))
    red_prior = 1 - white_prior

    return white_prior, red_prior


# 计算均值、方差
def get_mean_std(white_pixels, red_pixels):
    white_mean = np.mean(white_pixels)
    red_mean = np.mean(red_pixels)
    white_std = np.std(white_pixels)
    red_std = np.std(red_pixels)

    return white_mean, red_mean, white_std, red_std


# 计算均值、协方差矩阵
def get_mean_cov(white_pixels, red_pixels):
    white_mean = np.mean(white_pixels, axis=0)
    red_mean = np.mean(red_pixels, axis=0)
    white_cov = np.cov(white_pixels, rowvar=False)
    red_cov = np.cov(red_pixels, rowvar=False)

    return white_mean, red_mean, white_cov, red_cov


# 读取mask
def get_mask(mat_file, mat_name):
    mask_mat = io.loadmat(mat_file)[mat_name]

    return mask_mat


# 得到mask处理后的灰度图
def get_img_gray_masked(img_name, mask):
    img = cv2.imread(img_name)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray_masked = img_gray * mask / 255

    return img_gray_masked


# 得到mask处理后RGB图
def get_img_rgb_masked(img_name, mask):
    img = cv2.imread(img_name)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = np.array([mask, mask, mask]).transpose(1, 2, 0)
    img_rgb_masked = img_rgb * mask / 255

    return img_rgb_masked


# 分类灰度图像素点
def img_gray_segmentation(img_gray, priors, means, stds, weight):
    height, width = img_gray.shape[0], img_gray.shape[1]
    rtn_img_gray = np.zeros_like(img_gray)
    for i in range(height):
        for j in range(width):
            if img_gray[i][j] == 0:
                continue
            val_1 = priors[0] * norm.pdf(img_gray[i][j], means[0], stds[0])
            val_2 = priors[1] * norm.pdf(img_gray[i][j], means[1], stds[1])
            if weight * val_1 > val_2:
                rtn_img_gray[i][j] = 255
            else:
                rtn_img_gray[i][j] = 100

    return rtn_img_gray


# 分类RGB图像素点
def img_rgb_segmentation(img_rgb, priors, means, covs, weight):
    height, width = img_rgb.shape[0], img_rgb.shape[1]
    rtn_img_rgb = np.zeros_like(img_rgb)
    for i in range(height):
        for j in range(width):
            if img_rgb[i][j].sum() == 0:
                continue
            val_1 = priors[0] * multivariate_normal.pdf(img_rgb[i][j], means[0], covs[0])
            val_2 = priors[1] * multivariate_normal.pdf(img_rgb[i][j], means[1], covs[1])
            if weight * val_1 > val_2:
                rtn_img_rgb[i][j] = [255, 255, 255]
            else:
                rtn_img_rgb[i][j] = [255, 0, 0]

    return rtn_img_rgb


# 保存灰度图
def img_gray_save(save_dir, prefix, img_name, img_arr):
    img_to_save = save_dir + '/' + prefix + '_' + img_name
    cv2.imwrite(img_to_save, img_arr)

def img_rgb_save(save_dir, prefix, img_name, img_arr):
    img_to_save = save_dir + '/' + prefix + '_' + img_name
    img_arr = cv2.cvtColor(img_arr.astype(np.float32), cv2.COLOR_RGB2BGR)
    cv2.imwrite(img_to_save, img_arr)
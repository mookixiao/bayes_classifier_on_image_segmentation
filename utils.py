import cv2
import numpy as np
from scipy import io
from scipy.stats import norm, multivariate_normal

### task_1-1
# 读取Nemo鱼样本数据
def get_fish_sample(mat_file, mat_name):
    mat = io.loadmat(mat_file)

    gray_arr = mat[mat_name][:, 0]
    rgb_arr = mat[mat_name][:, 1:4]
    label_arr = mat[mat_name][:, -1]

    return gray_arr, rgb_arr, label_arr


# 分类像素点为白色、红色
def get_two_color_pixels(pixel_arr, label_arr):
    pixel_lst = pixel_arr.tolist()
    label_lst = label_arr.tolist()

    color_1_pixels = []
    color_2_pixels = []
    for i in range(len(pixel_lst)):
        if label_lst[i] == -1:
            color_1_pixels.append(pixel_lst[i])
        else:
            color_2_pixels.append(pixel_lst[i])
    color_1_pixels = np.array(color_1_pixels)
    color_2_pixels = np.array(color_2_pixels)

    return color_1_pixels, color_2_pixels


# 计算先验概率
def get_prior(arr_1, arr_2):
    prior_1 = len(arr_1) / (len(arr_2))
    prior_2 = 1 - prior_1

    return prior_1, prior_2


# 计算均值、方差
def get_mean_std(white_pixels, red_pixels):
    white_mean = np.mean(white_pixels)
    red_mean = np.mean(red_pixels)
    white_std = np.std(white_pixels)
    red_std = np.std(red_pixels)

    return white_mean, red_mean, white_std, red_std


# 计算均值、协方差矩阵
def get_mean_cov(color_1_pixels, color_2_pixels):
    color_1_mean = np.mean(color_1_pixels, axis=0)
    color_2_mean = np.mean(color_2_pixels, axis=0)
    color_1_cov = np.cov(color_1_pixels, rowvar=False)
    color_2_cov = np.cov(color_2_pixels, rowvar=False)

    return color_1_mean, color_2_mean, color_1_cov, color_2_cov


# 读取mask
def get_mask(mat_file, mat_name):
    mask_mat = io.loadmat(mat_file)[mat_name]

    return mask_mat


# 得到mask处理后的灰度图
def get_img_gray_masked(img, mask):
    img = cv2.imread(img)
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
def segment_img_gray(img, mask, priors, means, stds, weight=1):
    img_gray_masked = get_img_gray_masked(img, mask)
    rtn_img_gray = np.zeros_like(img_gray_masked)

    val_1 = priors[0] * norm.pdf(img_gray_masked, means[0], stds[0])
    val_2 = priors[1] * norm.pdf(img_gray_masked, means[1], stds[1])

    mask_idx = np.bool8(mask)
    color_1_idx = weight * val_1 > val_2

    rtn_img_gray[mask_idx] = 100
    rtn_img_gray[color_1_idx] = 255

    return rtn_img_gray


# 分类RGB图像素点
def segment_img_rgb(img, mask, priors, means, covs, weight=1):
    img_rgb_masked = get_img_rgb_masked(img, mask)
    rtn_img_rgb = np.zeros_like(img_rgb_masked)

    val_1 = priors[0] * multivariate_normal.pdf(img_rgb_masked, means[0], covs[0])
    val_2 = priors[1] * multivariate_normal.pdf(img_rgb_masked, means[1], covs[1])

    mask_idx = np.bool8(mask)
    color_1_idx = weight * val_1 > val_2

    rtn_img_rgb[mask_idx] = [255, 0, 0]
    rtn_img_rgb[color_1_idx] = [255, 255, 255]

    return rtn_img_rgb


# 保存灰度图
def img_gray_save(save_dir, prefix, img_name, img_arr):
    img_to_save = save_dir + '/' + prefix + '_' + img_name
    cv2.imwrite(img_to_save, img_arr)

# 保存RGB图
def img_rgb_save(save_dir, prefix, img_name, img_arr):
    img_to_save = save_dir + '/' + prefix + '_' + img_name
    img_arr = cv2.cvtColor(img_arr.astype(np.float32), cv2.COLOR_RGB2BGR)
    cv2.imwrite(img_to_save, img_arr)


### 以下针对task_1-2
# 读取整张样本数据
def get_whole_sample(img_file, mat_file, mat_name):
    img = cv2.imread(img_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask_mat = io.loadmat(mat_file)[mat_name]

    # 分类像素点
    fg_pixels = []
    bg_pixels = []
    img_lst = img.tolist()

    height, width = img.shape[0], img.shape[1]

    for i in range(height):
        for j in range(width):
            if mask_mat[i, j] == 1:
                fg_pixels.append(img_lst[i][j])
            else:
                bg_pixels.append(img_lst[i][j])

    fg_pixels = np.array(fg_pixels)
    bg_pixels = np.array(bg_pixels)

    return fg_pixels, bg_pixels

# 分类像素点为前景、背景
def get_fg_rbg_pixels(pixel_arr, label_arr):
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


# 获取每张图片mask并保存
def generate_mask(out_dir, img_dir, img_name, height_range, width_range, priors, means, covs, weight=1):
    print('GENERATE_MASK: the mask of ' + img_dir + '/' + img_name + ' generating...')

    img = cv2.imread(img_dir + '/' + img_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_mask = np.zeros(img.shape[:2])

    fg_prob = priors[0] * multivariate_normal.pdf(img, means[0], covs[0])
    bg_prob = priors[1] * multivariate_normal.pdf(img, means[1], covs[1])

    img_mask[weight * fg_prob > bg_prob] = 1
    img_mask[:height_range[0], :] = 0
    img_mask[height_range[1]:, :] = 0
    img_mask[:, :width_range[0]] = 0
    img_mask[:, width_range[1]:] = 0

    # 膨胀、腐蚀操作
    kernel = np.ones([5, 5])
    img_mask = cv2.dilate(img_mask, kernel)
    img_mask = cv2.erode(img_mask, kernel)

    mask_file = 'Mask_of_' + img_name + '.mat'
    mask_name = 'Mask'
    io.savemat(out_dir + '/' + mask_file, {mask_name: img_mask})

    print('GENERATE_MASK: the mask of ' + img_dir + '/' + img_name + ' saved as ' + out_dir + '/' + mask_file)
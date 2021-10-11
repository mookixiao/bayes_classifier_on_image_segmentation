# 基于无标签样本数据对309.bmp灰度图进行处理
import os

from utils import *


# 迭代开始
def img_rgb_segmentation_using_em(sample, img_rgb_masked, save_info, epochs, priors_init, means_init, covs_init):
    color_1_prior = priors_init[0]
    color_2_prior = priors_init[1]

    color_1_mean = means_init[0]
    color_2_mean = means_init[1]

    color_1_cov = covs_init[0]
    color_2_cov = covs_init[1]

    # EM算法
    labels = np.zeros((sample.shape[0], 2))
    color_1_rgb = np.zeros((sample.shape[0], 3))
    color_2_rgb = np.zeros((sample.shape[0], 3))
    for epoch in range(epochs):
        print('EM: Epoch_' + str(epoch + 1) + ' starting...')
        ### 第一步：E过程
        for i in range(len(sample)):
            color_1_val = color_1_prior * multivariate_normal.pdf(sample[i], color_1_mean, color_1_cov)
            color_2_val = color_2_prior * multivariate_normal.pdf(sample[i], color_2_mean, color_2_cov)

            color_1_label = color_1_val / (color_1_val + color_2_val)
            color_2_label = 1 - color_1_label

            labels[i, 0], labels[i, 1] = color_1_label, color_2_label  # 此数据的多少比例属于白色/红色
            color_1_rgb[i] = sample[i] * color_1_label  # 此数据的多少属于白色/红色
            color_2_rgb[i] = sample[i] * color_2_label

        ### 第二步：M过程
        # 更新先验概率
        color_1_cnt = labels[:, 0].sum()
        color_2_cnt = labels[:, 1].sum()
        color_1_prior = color_1_cnt / (color_1_cnt + color_2_cnt)
        color_2_prior = 1 - color_1_prior

        # 更新均值
        color_1_mean = color_1_rgb.sum(axis=0) / color_1_cnt
        color_2_mean = color_2_rgb.sum(axis=0) / color_2_cnt

        # 更新协方差矩阵
        color_1_cov_sum = np.zeros([3, 3])
        color_2_cov_sum = np.zeros([3, 3])
        for i in range(len(sample)):
            color_1_cov_sum = color_1_cov_sum + np.dot((sample[i] - color_1_mean).reshape(3, 1),
                                                       (sample[i] - color_1_mean).reshape(1, 3)) * labels[i, 0]
            color_2_cov_sum = color_2_cov_sum + np.dot((sample[i] - color_2_mean).reshape(3, 1),
                                                       (sample[i] - color_2_mean).reshape(1, 3)) * labels[i, 1]

        color_1_cov = color_1_cov_sum / (color_1_cnt - 1)  # 无偏估计除以N-1
        color_2_cov = color_2_cov_sum / (color_2_cnt - 1)

        ### 第三步：使用当前参数处理图像并保存
        img_rgb_processed = img_rgb_segmentation(img_rgb_masked,(color_1_prior, color_2_prior),
                                                 (color_1_mean, color_2_mean), (color_1_cov, color_2_cov))
        img_rgb_save(save_info[0], save_info[1] + 'Epoch-' + str(epoch), save_info[2], img_rgb_processed)


# 超参数
img_dir = 'imgs'
img_to_process = '309.bmp'

sample_mat_file = 'array_sample.mat'
sample_mat_name = 'array_sample'

mask_mat_file = 'Mask.mat'
mask_mat_name = 'Mask'

out_dir = 'out/task_2'

epochs = 30
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


if __name__ == '__main__':
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    _, rgb_arr, _ = get_fish_sample(sample_mat_file, sample_mat_name)
    mask = get_mask(mask_mat_file, mask_mat_name)
    img_masked = get_img_rgb_masked(img_dir + '/' + img_to_process, mask)
    # EM算法
    img_rgb_segmentation_using_em(rgb_arr, img_masked, (out_dir, 'EM', img_to_process), epochs,
                                  (white_prior, red_prior), (white_means, red_means), (white_cov, red_cov))
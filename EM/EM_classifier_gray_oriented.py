import numpy as np
import os
from PIL import Image
from scipy import io
from scipy.stats import norm
from EM.utils import save_gray_img, save_two_pdf_curves

# 输出文件夹
dir_out = 'EM_out'
if not os.path.isdir(dir_out):
    os.mkdir(dir_out)

# 超参
epochs = 30

# 训练数据
sample = io.loadmat('../array_sample.mat')['array_sample']
sample_gray = sample[:, 0]

# 待处理图片
img = Image.open('../imgs/309.bmp')
img_gray = np.array(img.convert('L'))
mask = io.loadmat('../Mask.mat')['Mask']
img_gray_masked = img_gray * mask / 255

save_gray_img(dir_out, 'img_gray_masked.png', img_gray_masked)

# 初始化先验概率
white_prior = 0.5
red_prior = 1 - white_prior

# 初始化均值、标准差
white_mean = 0.8
white_std = 0.3
red_mean = 0.5
red_std = 0.1

# 绘制初始pdf
save_two_pdf_curves(dir_out, 'pdf_curves_Origin.png', (white_mean, red_mean), (white_std, red_std), ('white', 'red'))

# 迭代开始
tmp = np.zeros((sample_gray.shape[0], 4))
for epoch in range(epochs):
    ### E过程
    for i in range(sample_gray.shape[0]):
        white_score = white_prior * norm.pdf(sample_gray[i], white_mean, white_std)
        red_score = red_prior * norm.pdf(sample_gray[i], red_mean, red_std)

        white_label = white_score / (white_score + red_score)
        red_label = 1 - white_label

        tmp[i, 0], tmp[i, 1] = white_label, red_label  # 此数据的多少比例属于白色/红色
        tmp[i, 2], tmp[i, 3] = white_label * sample_gray[i], red_label * sample_gray[i]  # 此数据的多少属于白色/红色

    ### M过程
    # 更新先验概率
    white_num = tmp[:, 0].sum()
    red_num = tmp[:, 1].sum()
    white_prior = white_num / (white_num + red_num)
    red_prior = 1 - white_prior

    # 更新均值
    white_mean = tmp[:, 2].sum() / white_num
    red_mean = tmp[:, 3].sum() / red_num

    # 更新标准差
    white_dif_sq_sum = (np.power((sample_gray - white_mean), 2) * tmp[:, 0]).sum()  #  差的平方的和
    red_dif_sq_sum = (np.power((sample_gray - red_mean), 2) * tmp[:, 1]).sum()
    white_std = np.sqrt(white_dif_sq_sum / white_num)
    red_std = np.sqrt(red_dif_sq_sum / red_num)

    # 绘制更新的pdf曲线
    save_two_pdf_curves(dir_out, 'pdf_curves_Epoch' + str(epoch + 1) + '.png',
                        (white_mean , red_mean), (white_std, red_std), ('white', 'red'))

    ### 使用当前参数处理图像
    img_out = np.zeros_like(img_gray_masked)
    for m in range(img_gray_masked.shape[0]):
        for n in range(img_gray_masked.shape[1]):
            if img_gray_masked[m][n] == 0:
                continue
            white_val = white_prior * norm.pdf(img_gray_masked[m][n], white_mean, white_std)
            red_val = red_prior * norm.pdf(img_gray_masked[m][n], red_mean, red_std)
            if white_val > red_val:
                img_out[m][n] = 1
            else:
                img_out[m][n] = 0.5

    save_gray_img(dir_out, 'img_processed_Epoch' + str(epoch + 1) + '.png', img_out)
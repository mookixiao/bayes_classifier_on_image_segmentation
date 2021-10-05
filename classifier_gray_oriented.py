import cv2
import numpy as np
from scipy import io
from scipy.stats import norm


# 读取数据
sample_mat = io.loadmat('array_sample.mat')
sample_gray_mat = sample_mat['array_sample'][:, 0]
sample_label_mat = sample_mat['array_sample'][:, -1]
sample_gray = sample_gray_mat.tolist()
sample_label = sample_label_mat.tolist()

white_pixes = []
red_pixes = []
for i in range(len(sample_gray)):
    if sample_label[i] == -1:
        white_pixes.append(sample_gray[i])
    else:
        red_pixes.append(sample_gray[i])
white_pixes = np.array(white_pixes)
red_pixes = np.array(red_pixes)

# 计算先验概率
white_prior = len(white_pixes) / (len(sample_gray))
red_prior = len(red_pixes) / len(sample_gray)

# 计算均值、方差
white_mean = np.mean(white_pixes)
red_mean = np.mean(red_pixes)
white_var = np.var(white_pixes)
red_var = np.var(red_pixes)

# 读入待分类图片
mask_mat = io.loadmat('Mask.mat')
mask_mat = mask_mat['Mask']
img = cv2.imread('imgs/309.bmp')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray_masked = img_gray * mask_mat / 255
img_gray_processed = np.array(img_gray_masked)

def classifier(weight):
    # 二分类
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img_gray_masked[i][j] == 0:
                continue
            white_val = white_prior * norm.pdf(img_gray_masked[i][j], white_mean, np.sqrt(white_var))
            red_val = red_prior * norm.pdf(img_gray_masked[i][j], red_mean, np.sqrt(red_var))
            if weight * white_val > red_val:
                img_gray_processed[i][j] = 0  # 此处应为1和0，为明显起见，反转两者
            else:
                img_gray_processed[i][j] = 1

    cv2.imshow('BGR', img)
    cv2.imshow('gray', img_gray)
    cv2.imshow('gray masked', img_gray_masked)
    cv2.imshow('gray processed, W=' + str(weight), img_gray_processed)

if __name__ == '__main__':
    classifier(weight=0.4)  # 当权重为0.4时效果较好

    cv2.waitKey()
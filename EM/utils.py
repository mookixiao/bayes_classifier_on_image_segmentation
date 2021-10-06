import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.stats import norm


def save_gray_img(dir, name, gray_img):
    if gray_img.max() <= 1:
        gray_img = (gray_img * 255).round()
    img = Image.fromarray(gray_img)
    img = img.convert('RGB')
    img.save(dir + '/' + name)


def save_two_pdf_curves(dir, name, means, stds, labels):
    x = np.arange(0, 1, 1 / 1000)
    pdf_a = norm.pdf(x, means[0], stds[0])
    pdf_b = norm.pdf(x, means[1], stds[1])

    ax = plt.subplot(1, 1, 1)
    ax.set_title(name)
    ax.plot(x, pdf_a, 'r', label=labels[0])
    ax.plot(x, pdf_b, 'b', label=labels[1])
    plt.savefig(dir + '/' + name)
    plt.close()
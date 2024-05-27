import cv2
import numpy as np
import matplotlib.pyplot as plt

def quantize_histogram(histogram, bin_size=5):
    # 量化直方图
    num_bins = 180 // bin_size  # 180 是H通道的最大值
    quantized_histogram = np.zeros((num_bins,))

    # 遍历原始直方图，将值累加到量化直方图的相应桶中
    for i in range(len(histogram)):
        bin_index = i // bin_size
        quantized_histogram[bin_index] += histogram[i][0]

    return quantized_histogram


def plot_histograms(image, mask):
    # 获取H通道
    hue_channel = image[:, :, 0]
    # 使用掩码计算直方图，只对前景部分进行计算
    histogram = cv2.calcHist([hue_channel], [0], mask, [180], [0, 180])  # 注意这里的bins参数改为180

    # 量化直方图
    quantized_histogram = quantize_histogram(histogram, bin_size=5)

    # 归一化量化直方图
    total_pixels = sum(quantized_histogram)
    normalized_quantized_histogram = quantized_histogram / total_pixels

    # 显示三个直方图
    plt.figure(figsize=(15, 5))

    # 绘制未量化的直方图
    plt.subplot(1, 3, 1)
    plt.plot(histogram, color='b')
    plt.title('Original Histogram of H Channel')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')

    # 绘制量化后的直方图
    plt.subplot(1, 3, 2)
    plt.bar(range(len(quantized_histogram)), quantized_histogram, color='b')
    plt.title('Quantized Histogram of H Channel')
    plt.xlabel('Quantized H Value')
    plt.ylabel('Frequency')

    # 绘制归一化的量化直方图
    plt.subplot(1, 3, 3)
    plt.bar(range(len(normalized_quantized_histogram)), normalized_quantized_histogram, color='b')
    plt.title('Normalized Quantized Histogram of H Channel')
    plt.xlabel('Quantized H Value')
    plt.ylabel('Color Proportion')

    plt.tight_layout()
    plt.show()

    return normalized_quantized_histogram


def find_main_color(histogram, threshold=0.500):
    # 找到主要颜色分量及其索引
    max_index = np.argmax(histogram)
    main_color_proportion = histogram[max_index]

    if max_index > 0:
        main_color_proportion += histogram[max_index - 1]

    if max_index < len(histogram) - 1:
        main_color_proportion += histogram[max_index + 1]

    return main_color_proportion < threshold




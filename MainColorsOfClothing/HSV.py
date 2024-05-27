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


def find_main_color(normalized_quantized_histogram, threshold=0.65):
    # 找到主要颜色分量及其索引
    max_index = np.argmax(normalized_quantized_histogram)
    main_color_proportion = normalized_quantized_histogram[max_index]

    # 计算主要颜色占比及左右各1的颜色分量的占比之和
    if max_index > 0:
        main_color_proportion += normalized_quantized_histogram[max_index - 1]
        normalized_quantized_histogram[max_index - 1] = 0  # 将左边颜色分量占比置为0

    if max_index < len(normalized_quantized_histogram) - 1:
        main_color_proportion += normalized_quantized_histogram[max_index + 1]
        normalized_quantized_histogram[max_index + 1] = 0  # 将右边颜色分量占比置为0

    # 更新主颜色位置的值为新的占比值
    normalized_quantized_histogram[max_index] = main_color_proportion

    # 创建Flag数组，并存储主颜色特征值
    Flag = [0]  # 初始时只有一个元素，表示长度

    # 添加主颜色索引和占比到Flag数组
    Flag.append(max_index)
    Flag.append(main_color_proportion)

    # 更新Flag数组的第一个元素为当前数组长度
    Flag[0] = len(Flag) - 1  # 第一个元素不计入长度

    # 打印Flag数组和其中的特征值
    print("Flag数组:", Flag)
    print("Flag数组长度:", Flag[0])
    print("主要颜色索引:", Flag[1])
    print("主要颜色比例:", Flag[2])



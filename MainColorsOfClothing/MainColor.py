import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

from CutOut import (
    determine_background_complexity,
    generate_initial_rect_simple,
    generate_initial_rect_complex,
    apply_grabcut,
    extract_background_from_s,
    extract_foreground_from_x
)

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

    # 显示未量化的直方图、量化直方图和归一化的量化直方图
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

    return Flag

# 读取并显示所有裁剪后的图片，按q键退出
# 使用原始字符串定义标注文件路径
annotation_file_path = r'D:/tuxiangchuli/Anno_coarse/list_bbox.txt'

# 确保文件存在
if os.path.exists(annotation_file_path):
    with open(annotation_file_path, 'r') as file:
        # 跳过前两行，因为它们不是标注数据
        next(file)
        next(file)

        # 逐行读取标注数据
        for line in iter(file):
            parts = line.strip().split()
            if len(parts) == 5:  # 确保行数据长度正确
                image_name = parts[0]
                x1, y1, x2, y2 = map(int, parts[1:])

                # 确保标注坐标在合理范围内
                if all(0 <= coord <= 32767 for coord in (x1, y1, x2, y2)) and x2 > x1 and y2 > y1:
                    image_path = 'D:/tuxiangchuli' + '/' + image_name  # 构造图片路径
                    # 读取图片并检查是否成功读取
                    image = cv2.imread(image_path)
                    if image is not None:
                        # 裁剪图片
                        cropped_image = image[y1:y2, x1:x2]

                        # 判断背景复杂度
                        complexity = determine_background_complexity(cropped_image)
                        if complexity == "simple":
                            rect = generate_initial_rect_simple(cropped_image)
                        else:
                            rect = generate_initial_rect_complex(cropped_image)


                        # 应用GrabCut算法并生成S图
                        result, s_image = apply_grabcut(cropped_image, rect)

                        # 提取背景X图
                        x_image = extract_background_from_s(s_image)

                        # 提取最终的完整前景图
                        final_foreground = extract_foreground_from_x(cropped_image, x_image)

                        cv2.namedWindow("Final Foreground", cv2.WINDOW_NORMAL)
                        cv2.imshow("Final Foreground", final_foreground)

                        # 将图片从RGB颜色空间转换为HSV颜色空间
                        hsv_image = cv2.cvtColor(final_foreground, cv2.COLOR_BGR2HSV)

                        cv2.namedWindow("HSV Image", cv2.WINDOW_NORMAL)
                        cv2.imshow("HSV Image", hsv_image)

                        # 创建掩码，前景像素为1，背景像素为0
                        foreground_mask = (final_foreground[:, :, 0] != 0).astype(np.uint8)

                        # 绘制直方图和找到主要颜色
                        normalized_quantized_histogram = plot_histograms(hsv_image, foreground_mask)
                        Flag = find_main_color(normalized_quantized_histogram)

                        # 等待用户按键操作，如果按下 'q' 则退出循环
                        if cv2.waitKey(0) & 0xFF == ord('q'):
                            cv2.destroyAllWindows()
                            break
                    else:
                        print("图片读取失败，请检查图片路径或文件格式。")
                else:
                    print("标注坐标无效或不完整。")
        else:  # 没有遇到break，即所有图片都显示完毕
            print("所有图片显示完毕。")
else:
    print("标注文件不存在，请检查文件路径是否正确。")

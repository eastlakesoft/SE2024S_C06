import cv2
import numpy as np
import os
import CutOut
import HSV
import MainColor
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 主函数，处理用户上传的图片
def process_uploaded_image(image_path):
    image = cv2.imread(image_path)
    if image is not None:
        cropped_image = image  # 在这里我们使用整个图像作为裁剪后的图像

        complexity = CutOut.determine_background_complexity(cropped_image)
        if complexity == "simple":
            rect = CutOut.generate_initial_rect_simple(cropped_image)
        else:
            rect = CutOut.generate_initial_rect_complex(cropped_image)

        debug_image = cropped_image.copy()
        cv2.rectangle(debug_image, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
        cv2.imshow("Initial Rect", debug_image)

        result, s_image = CutOut.apply_grabcut(cropped_image, rect)

        cv2.namedWindow("Cropped Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Cropped Image", cropped_image)

        # cv2.namedWindow("GrabCut Result", cv2.WINDOW_NORMAL)
        # cv2.imshow("GrabCut Result", result)
        #
        # cv2.namedWindow("S Image", cv2.WINDOW_NORMAL)
        # cv2.imshow("S Image", s_image)
        #
        x_image = CutOut.extract_background_from_s(s_image)
        #
        # cv2.namedWindow("X Image", cv2.WINDOW_NORMAL)
        # cv2.imshow("X Image", x_image)

        final_foreground = CutOut.extract_foreground_from_x(cropped_image, x_image)

        cv2.namedWindow("Final Foreground", cv2.WINDOW_NORMAL)
        cv2.imshow("Final Foreground", final_foreground)

        # 判断抠图是否成功（根据前景图像中有效像素的比例）
        foreground_pixels = np.count_nonzero(final_foreground)
        total_pixels = cropped_image.shape[0] * cropped_image.shape[1]
        foreground_ratio = foreground_pixels / total_pixels

        if foreground_ratio < 0.1:  # 抠图失败，使用原图
            dominant_colors = MainColor.extract_dominant_colors(cropped_image)
            hsv_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
            foreground_mask = (cropped_image[:, :, 0] != 0).astype(np.uint8)
            cv2.namedWindow("Image Used for Color Extraction", cv2.WINDOW_NORMAL)
            cv2.imshow("Image Used for Color Extraction", cropped_image)  # 显示用于颜色提取的图片（原图）
        else:  # 抠图成功，使用抠出的前景图
            dominant_colors = MainColor.extract_dominant_colors(final_foreground)
            hsv_image = cv2.cvtColor(final_foreground, cv2.COLOR_BGR2HSV)
            foreground_mask = (final_foreground[:, :, 0] != 0).astype(np.uint8)
            cv2.namedWindow("Image Used for Color Extraction", cv2.WINDOW_NORMAL)
            cv2.imshow("Image Used for Color Extraction", final_foreground)  # 显示用于颜色提取的图片（前景图）

        normalized_quantized_histogram = HSV.plot_histograms(hsv_image, foreground_mask)

        # 判断是否为花色服装
        is_flower = MainColor.is_multicolor(dominant_colors, threshold=70) or HSV.find_main_color(
            normalized_quantized_histogram, threshold=0.500)

        if is_flower:
            MainColor.display_colors(dominant_colors, title="花色服装: 前两种颜色")
        else:
            MainColor.display_colors([dominant_colors[0]], title="纯色服装: 主颜色")

        # 确保所有窗口都关闭，以便plt显示图表和主颜色
        if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
    else:
        print("图片读取失败，请检查图片路径或文件格式。")

# 用户上传图片的路径
uploaded_image_path = '用户上传的图片路径'
process_uploaded_image(uploaded_image_path)
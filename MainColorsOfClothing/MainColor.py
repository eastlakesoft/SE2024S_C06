import cv2
import numpy as np
import os
import CutOut
import HSV
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 去除无效颜色yo
def remove_invalid_colors(colors, threshold=15):
    mask = np.all(colors > threshold, axis=1) & np.all(colors < 255 - threshold, axis=1)
    return colors[mask]


# 提取图像的主要颜色
def extract_dominant_colors(image, k=3):
    pixels = image.reshape((-1, 3))
    pixels = remove_invalid_colors(pixels)

    if len(pixels) == 0:
        return [[0, 0, 0], [0, 0, 0]]  # 如果没有有效像素，返回黑色

    pixels = np.float32(pixels)
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    _, counts = np.unique(labels, return_counts=True)
    dominant_colors = centers[np.argsort(counts)[-2:]]  # 选择占比最大的两个颜色

    return dominant_colors.astype(int)


# 显示颜色
def display_colors(colors, title="Colors"):
    num_colors = len(colors)
    fig, ax = plt.subplots(1, num_colors, figsize=(num_colors * 2, 2))
    if num_colors == 1:
        color_img = np.zeros((50, 50, 3), dtype=np.uint8)
        color_img[:, :] = colors[0]
        ax.imshow(color_img[..., ::-1])  # 将BGR转换为RGB进行显示
        ax.axis('off')
    else:
        for i, color in enumerate(colors):
            color_img = np.zeros((50, 50, 3), dtype=np.uint8)
            color_img[:, :] = color
            ax[i].imshow(color_img[..., ::-1])  # 将BGR转换为RGB进行显示
            ax[i].axis('off')
    plt.suptitle(title)
    plt.show()


# 判断是否为花色服装
def is_multicolor(dominant_colors, threshold=50):
    color_distance = np.linalg.norm(dominant_colors[0] - dominant_colors[1])
    return color_distance > threshold


annotation_file_path = r'D:/tuxiangchuli/Anno_coarse/list_bbox.txt'

if os.path.exists(annotation_file_path):
    with open(annotation_file_path, 'r') as file:
        next(file)
        next(file)

        for line in iter(file):
            parts = line.strip().split()
            if len(parts) == 5:
                image_name = parts[0]
                x1, y1, x2, y2 = map(int, parts[1:])

                if all(0 <= coord <= 32767 for coord in (x1, y1, x2, y2)) and x2 > x1 and y2 > y1:
                    image_path = 'D:/tuxiangchuli' + '/' + image_name
                    image = cv2.imread(image_path)
                    if image is not None:
                        cropped_image = image[y1:y2, x1:x2]

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

                        cv2.namedWindow("GrabCut Result", cv2.WINDOW_NORMAL)
                        cv2.imshow("GrabCut Result", result)

                        cv2.namedWindow("S Image", cv2.WINDOW_NORMAL)
                        cv2.imshow("S Image", s_image)

                        x_image = CutOut.extract_background_from_s(s_image)

                        cv2.namedWindow("X Image", cv2.WINDOW_NORMAL)
                        cv2.imshow("X Image", x_image)

                        final_foreground = CutOut.extract_foreground_from_x(cropped_image, x_image)

                        cv2.namedWindow("Final Foreground", cv2.WINDOW_NORMAL)
                        cv2.imshow("Final Foreground", final_foreground)

                        dominant_colors = extract_dominant_colors(final_foreground)
                        hsv_image = cv2.cvtColor(final_foreground, cv2.COLOR_BGR2HSV)
                        foreground_mask = (final_foreground[:, :, 0] != 0).astype(np.uint8)
                        normalized_quantized_histogram = HSV.plot_histograms(hsv_image, foreground_mask)

                        # 判断是否为花色服装
                        is_flower = is_multicolor(dominant_colors,
                                                  threshold=70) or HSV.find_main_color(
                            normalized_quantized_histogram, threshold=0.550)

                        if is_flower:
                            display_colors(dominant_colors, title="Flower Clothing: Top 2 Colors")
                        else:
                            display_colors([dominant_colors[0]], title="Solid Clothing: Main Color")

                        if cv2.waitKey(0) & 0xFF == ord('q'):
                            cv2.destroyAllWindows()
                            break
                    else:
                        print("图片读取失败，请检查图片路径或文件格式。")
                else:
                    print("标注坐标无效或不完整。")
        else:
            print("所有图片显示完毕。")
else:
    print("标注文件不存在，请检查文件路径是否正确。")

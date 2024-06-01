import cv2
import numpy as np
import os

from matplotlib import pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans


def color_distance(c1, c2):
    # 计算两个颜色之间的欧氏距离
    return np.sqrt(np.sum((c1 - c2) ** 2))


def extract_corner_colors(image):
    # 提取图像四个角点的颜色特征
    height, width = image.shape[:2]
    corners = [
        image[0, 0],  # 左上角
        image[0, width - 1],  # 右上角
        image[height - 1, 0],  # 左下角
        image[height - 1, width - 1]  # 右下角
    ]
    return np.array(corners)


def determine_background_complexity(image):
    # 判断图像背景的复杂度
    corners = extract_corner_colors(image)
    distances = [
        color_distance(corners[i], corners[j])
        for i in range(len(corners)) for j in range(i + 1, len(corners))
    ]

    count_lt_5 = sum(d < 5 for d in distances)  # 统计欧氏距离小于5的个数
    count_lt_55 = sum(d < 55 for d in distances)  # 统计欧氏距离小于55的个数

    # 根据统计结果判断背景复杂度
    return "simple" if count_lt_5 > 1 or count_lt_55 == 6 else "complex"


def generate_initial_rect_simple(image):
    # 生成单一背景图像的初始矩形框
    height, width = image.shape[:2]
    rect_width = int(width * 0.8)
    rect_height = int(height * 0.8)
    x = (width - rect_width) // 2
    y = (height - rect_height) // 2
    return (x, y, x + rect_width, y + rect_height)


def region_growing(image, seed_points):
    # 区域生长算法
    height, width = image.shape[:2]
    mask = np.zeros((height, width), np.uint8)

    for seed in seed_points:
        if 0 <= seed[0] < width and 0 <= seed[1] < height and image[seed[1], seed[0]] == 0:  # 只对黑色像素进行生长
            mask[seed[1], seed[0]] = 255
            stack = [seed]
            while stack:
                x, y = stack.pop()
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height and mask[ny, nx] == 0 and image[ny, nx] == 0:
                        mask[ny, nx] = 255
                        stack.append((nx, ny))

    return mask


def generate_initial_rect_complex(image):
    # 生成复杂背景图像的初始矩形框
    height, width, _ = image.shape
    block_size = height // 7
    center_block = image[3 * block_size:4 * block_size, 3 * block_size:4 * block_size]
    center_hist = cv2.calcHist([center_block], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])

    min_distance = float('inf')
    for y in [0, 6]:
        for x in [0, 6]:
            corner_block = image[y * block_size:(y + 1) * block_size, x * block_size:(x + 1) * block_size]
            corner_hist = cv2.calcHist([corner_block], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            distance = cv2.compareHist(center_hist, corner_hist, cv2.HISTCMP_CORREL)
            if distance < min_distance:
                min_distance = distance

    seed = (width // 2, height // 2)
    mask = region_growing(image, [seed])
    x, y, w, h = cv2.boundingRect(mask)
    return (x, y, x + w, y + h)


def apply_grabcut(image, rect):
    # 应用GrabCut算法提取前景
    mask = np.zeros(image.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # 初始化GrabCut算法
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    # 提取前景部分
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    image_result = image * mask2[:, :, np.newaxis]

    # 创建S图（初步前景图）
    s_image = np.where(mask2 == 1, 255, 0).astype('uint8')

    return image_result, s_image


def extract_background_from_s(s_image):
    # 提取S图的最外层背景X图
    height, width = s_image.shape[:2]
    seed_points = [
        (0, 0), (0, width // 2), (0, width - 1),
        (height // 2, 0), (height // 2, width - 1),
        (height - 1, 0), (height - 1, width // 2), (height - 1, width - 1)
    ]

    x_image = region_growing(s_image, seed_points)
    return x_image


def extract_foreground_from_x(image, x_image):
    # 将X图与原图进行与运算，获得最终的完整前景图
    x_mask = (x_image == 0).astype(np.uint8)  # 将背景掩码取反，前景为1，背景为0
    final_foreground = cv2.bitwise_and(image, image, mask=x_mask)
    return final_foreground


# 显示颜色
def display_colors(colors, title="颜色", filename="main_color.png"):
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
    # plt.show()
    plt.savefig("temp_plot.png", bbox_inches='tight', pad_inches=0)
    plt.close()

    # 使用Pillow保存截图
    img = Image.open("temp_plot.png")
    img.save(filename)
    os.remove("temp_plot.png")  # 删除临时文件# 显示颜色

# 去除无效颜色
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

# 判断是否为花色服装的函数
def is_multicolor(dominant_colors, threshold=50):

    color_distance = np.linalg.norm(dominant_colors[0] - dominant_colors[1])  # 计算颜色之间的欧式距离
    return color_distance > threshold  # 如果距离大于阈值，则返回True，否则返回False

# 去除无效颜色
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
def display_colors(colors, title="颜色", filename="main_color.png"):
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
    plt.savefig("temp_plot.png", bbox_inches='tight', pad_inches=0)
    plt.close()

    # 使用Pillow保存截图
    img = Image.open("temp_plot.png")
    img.save(filename)
    os.remove("temp_plot.png")  # 删除临时文件




# 判断是否为花色服装
def is_multicolor(dominant_colors, threshold=50):#
    color_distance = np.linalg.norm(dominant_colors[0] - dominant_colors[1])
    return color_distance > threshold

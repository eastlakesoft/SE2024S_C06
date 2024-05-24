import cv2
import numpy as np
import os


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

                        # 显示矩形框
                        debug_image = cropped_image.copy()
                        cv2.rectangle(debug_image, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
                        cv2.imshow("Initial Rect", debug_image)

                        # 应用GrabCut算法并生成S图
                        result, s_image = apply_grabcut(cropped_image, rect)

                        # 显示结果图像和S图
                        cv2.namedWindow("Cropped Image", cv2.WINDOW_NORMAL)
                        cv2.imshow("Cropped Image", cropped_image)

                        cv2.namedWindow("GrabCut Result", cv2.WINDOW_NORMAL)
                        cv2.imshow("GrabCut Result", result)

                        cv2.namedWindow("S Image", cv2.WINDOW_NORMAL)
                        cv2.imshow("S Image", s_image)

                        # 提取背景X图
                        x_image = extract_background_from_s(s_image)

                        cv2.namedWindow("X Image", cv2.WINDOW_NORMAL)
                        cv2.imshow("X Image", x_image)

                        # 提取最终的完整前景图
                        final_foreground = extract_foreground_from_x(cropped_image, x_image)

                        cv2.namedWindow("Final Foreground", cv2.WINDOW_NORMAL)
                        cv2.imshow("Final Foreground", final_foreground)

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

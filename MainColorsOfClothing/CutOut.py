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
    rect_width = 15
    rect_height = 15
    x = (width - rect_width) // 2
    y = (height - rect_height) // 2
    return (x, y, x + rect_width, y + rect_height)

def region_growing(image, seed, threshold):
    # 区域生长算法
    height, width = image.shape[:2]
    mask = np.zeros((height, width), np.uint8)
    mask[seed[1], seed[0]] = 255
    kernel = np.ones((3, 3), np.uint8)
    
    while True:
        dilated = cv2.dilate(mask, kernel, iterations=1)
        new_mask = cv2.absdiff(dilated, mask)
        mask = cv2.add(mask, new_mask)
        
        diff = cv2.absdiff(image, cv2.bitwise_and(image, image, mask=new_mask))
        mask_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        new_mask[mask_diff > threshold] = 0
        
        if cv2.countNonZero(new_mask) == 0:
            break
    
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
    mask = region_growing(image, seed, min_distance)
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

def process_images_from_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            if image is not None:
                # 判断背景复杂度
                complexity = determine_background_complexity(image)
                if complexity == "simple":
                    rect = generate_initial_rect_simple(image)
                else:
                    rect = generate_initial_rect_complex(image)
                
                # 显示矩形框
                debug_image = image.copy()
                cv2.rectangle(debug_image, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
                cv2.imshow("Initial Rect", debug_image)
                
                # 应用GrabCut算法并生成S图
                result, s_image = apply_grabcut(image, rect)
                
                # 显示结果图像和S图
                cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
                cv2.imshow("Original Image", image)
                
                cv2.namedWindow("GrabCut Result", cv2.WINDOW_NORMAL)
                cv2.imshow("GrabCut Result", result)
                
                cv2.namedWindow("S Image", cv2.WINDOW_NORMAL)
                cv2.imshow("S Image", s_image)
                
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break
            else:
                print(f"Failed to read image: {image_path}")
    cv2.destroyAllWindows()

# 读取文件夹中的所有图片并进行处理
image_folder_path = r'D:\tuxiangchuli\img\2-in-1_Space_Dye_Athletic_Tank'
process_images_from_folder(image_folder_path)

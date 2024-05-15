import cv2
import numpy as np
import sys
import math
import os

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
                    # print(f"生成的图片路径: {image_path}")

                    # 读取图片并检查是否成功读取
                    image = cv2.imread(image_path)
                    if image is not None:
                        # 裁剪图片
                        cropped_image = image[y1:y2, x1:x2]

                        # 创建窗口并显示裁剪后的图片
                        cv2.namedWindow("Cropped Image", cv2.WINDOW_NORMAL)
                        cv2.imshow("Cropped Image", cropped_image)

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
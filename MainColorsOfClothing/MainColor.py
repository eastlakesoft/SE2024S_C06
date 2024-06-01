import cv2
import numpy as np
import os
import CutOut
import HSV


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

                        if foreground_ratio < 0.8: # 抠图失败，使用原图
                            dominant_colors = CutOut.extract_dominant_colors(cropped_image)
                            hsv_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
                            foreground_mask = (cropped_image[:, :, 0] != 0).astype(np.uint8)
                            cv2.namedWindow("Image Used for Color Extraction", cv2.WINDOW_NORMAL)
                            cv2.imshow("Image Used for Color Extraction", cropped_image)  # 显示用于颜色提取的图片（原图）
                        else:  # 抠图成功，使用抠出的前景图
                            dominant_colors = CutOut.extract_dominant_colors(final_foreground)
                            hsv_image = cv2.cvtColor(final_foreground, cv2.COLOR_BGR2HSV)
                            foreground_mask = (final_foreground[:, :, 0] != 0).astype(np.uint8)
                            cv2.namedWindow("Image Used for Color Extraction", cv2.WINDOW_NORMAL)
                            cv2.imshow("Image Used for Color Extraction", final_foreground)  # 显示用于颜色提取的图片（前景图）

                        normalized_quantized_histogram = HSV.plot_histograms(hsv_image, foreground_mask)

                        # 判断是否为花色服装
                        is_flower = CutOut.is_multicolor(dominant_colors, threshold=70) or HSV.find_main_color(
                            normalized_quantized_histogram, threshold=0.500)

                        if is_flower:
                            CutOut.display_colors(dominant_colors, title="Flower Clothing: Top 2 Colors")
                        else:
                            CutOut.display_colors([dominant_colors[0]], title="Solid Clothing: Main Color")

                        # 确保所有窗口都关闭，以便plt显示图表和主颜色
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

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import cv2
import os

from MainColorsOfClothing import HSV, CutOut




class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(2000, 1400)  # 调整主窗口大小为原来的2倍
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # 设置背景图
        self.background_label = QtWidgets.QLabel(self.centralwidget)
        self.background_label.setGeometry(0, 0, MainWindow.width(), MainWindow.height())
        self.background_label.setPixmap(
            QtGui.QPixmap("D:/计算机图像处理/320.jpg").scaled(MainWindow.size(), QtCore.Qt.IgnoreAspectRatio,
                                                              QtCore.Qt.SmoothTransformation))
        self.background_label.setScaledContents(True)

        # 添加标题
        self.title = QtWidgets.QLabel(self.centralwidget)
        self.title.setGeometry(QtCore.QRect(300, 80, 1400, 100))  # 调整标题位置和大小
        self.title.setObjectName("title")
        font = QtGui.QFont()
        font.setPointSize(48)  # 调整字体大小
        font.setBold(True)
        self.title.setFont(font)
        self.title.setAlignment(QtCore.Qt.AlignCenter)



        # 彩色标题文本
        title_text = "服装提取主色系统"
        title_colored = ''.join([f'<span style="color:#006400;">{char}</span>' for char in title_text])
        self.title.setText(title_colored)
        self.title.setStyleSheet("QLabel {color: #006400;}")

        # 添加美观框架
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(100, 200, 1800, 1200))  # 调整框架位置和大小
        self.frame.setFrameShape(QtWidgets.QFrame.Box)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setLineWidth(6)  # 调整边框宽度
        self.frame.setObjectName("frame")
        self.frame.setStyleSheet(
            "QFrame { border: none; background-color: rgba(255, 255, 255, 180); border-radius: 20px;}")

        self.pushButton = QtWidgets.QPushButton(self.frame)
        self.pushButton.setGeometry(QtCore.QRect(200, 1100, 200, 60))  # 调整按钮位置和大小
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.upload_image)

        self.pushButton_2 = QtWidgets.QPushButton(self.frame)
        self.pushButton_2.setGeometry(QtCore.QRect(1000, 1100, 200, 60))  # 调整按钮位置和大小
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.clicked.connect(self.process_image)

        self.label = QtWidgets.QLabel(self.frame)
        self.label.setGeometry(QtCore.QRect(100, 100, 600, 400))  # 调整标签位置和大小，使其占用上半部分

        self.label.setFrameShape(QtWidgets.QFrame.Box)
        self.label.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.label.setLineWidth(4)  # 调整边框宽度
        self.label.setObjectName("label")

        self.label_text = QtWidgets.QLabel(self.frame)
        self.label_text.setGeometry(QtCore.QRect(100, 500, 400, 40))  # 调整标签文本位置和大小
        self.label_text.setObjectName("label_text")

        self.label_2 = QtWidgets.QLabel(self.frame)
        self.label_2.setGeometry(QtCore.QRect(900, 100, 600, 400))  # 调整标签位置和大小
        self.label_2.setFrameShape(QtWidgets.QFrame.Box)
        self.label_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.label_2.setLineWidth(4)  # 调整边框宽度
        self.label_2.setObjectName("label_2")
        self.processed_image = None

        self.label_2_text = QtWidgets.QLabel(self.frame)
        self.label_2_text.setGeometry(QtCore.QRect(900, 500, 400, 40))  # 调整标签文本位置和大小
        self.label_2_text.setObjectName("label_2_text")

        self.label_3 = QtWidgets.QLabel(self.frame)

        self.label_3.setGeometry(QtCore.QRect(100, 600, 1000, 400))  # 调整标签位置和大小，使其占用下半部分的左侧


        self.label_3.setFrameShape(QtWidgets.QFrame.Box)
        self.label_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.label_3.setLineWidth(4)  # 调整边框宽度
        self.label_3.setObjectName("label_3")

        self.label_3_text = QtWidgets.QLabel(self.frame)
        self.label_3_text.setGeometry(QtCore.QRect(100, 1000, 400, 40))  # 调整标签文本位置和大小
        self.label_3_text.setObjectName("label_3_text")

        self.label_4 = QtWidgets.QLabel(self.frame)
        self.label_4.setGeometry(QtCore.QRect(1300, 600, 200, 400))  # 调整标签位置和大小，使其占用下半部分的右侧
        self.label_4.setFrameShape(QtWidgets.QFrame.Box)
        self.label_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.label_4.setLineWidth(4)  # 调整边框宽度
        self.label_4.setObjectName("label_4")

        self.label_4_text = QtWidgets.QLabel(self.frame)
        self.label_4_text.setGeometry(QtCore.QRect(1300, 1000, 200, 40))  # 调整标签文本位置和大小
        self.label_4_text.setObjectName("label_4_text")

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 2000, 44))  # 调整菜单栏大小
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # 初始化图片路径
        self.image_path = None

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "服装提取主色系统"))
        self.pushButton.setText(_translate("MainWindow", "上传图片"))
        self.pushButton_2.setText(_translate("MainWindow", "处理图片"))
        self.label.setText(_translate("MainWindow", ""))
        self.label_text.setText(_translate("MainWindow", "上传图片"))
        self.label_2.setText(_translate("MainWindow", ""))
        self.label_2_text.setText(_translate("MainWindow", "提取主颜色的图片"))
        self.label_3.setText(_translate("MainWindow", ""))
        self.label_3_text.setText(_translate("MainWindow", "HSV直方图"))
        self.label_4.setText(_translate("MainWindow", ""))
        self.label_4_text.setText(_translate("MainWindow", "主颜色"))

    def resizeEvent(self, event):
        self.background_label.setGeometry(0, 0, self.centralwidget.width(), self.centralwidget.height())
        self.background_label.setPixmap(
            QtGui.QPixmap("D:/计算机图像处理/320.jpg").scaled(self.centralwidget.size(), QtCore.Qt.IgnoreAspectRatio,
                                                              QtCore.Qt.SmoothTransformation))
        self.background_label.setScaledContents(True)

    def upload_image(self):
        options = QtWidgets.QFileDialog.Options()
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "选择图片", "",
                                                            "Image Files (*.png *.jpg *.jpeg *.bmp)", options=options)
        if fileName:
            self.image_path = fileName
            pixmap = QtGui.QPixmap(fileName)
            self.label.setPixmap(pixmap.scaled(self.label.size(), QtCore.Qt.KeepAspectRatio))



    def process_image(self):

        if self.image_path:
            # 调用之前的图像处理函数，并返回最终的前景图像
            final_foreground = self.process_uploaded_image(self.image_path)

            # 将前景图像转换为Qt格式并显示在label_2中
            height, width, channel = final_foreground.shape
            bytes_per_line = 3 * width
            q_img = QtGui.QImage(final_foreground.data, width, height, bytes_per_line,
                                 QtGui.QImage.Format_RGB888).rgbSwapped()
            self.label_2.setPixmap(
                QtGui.QPixmap.fromImage(q_img).scaled(self.label_2.size(), QtCore.Qt.KeepAspectRatio))

            # 调用plot_histograms并绘制直方图
            hsv_image = cv2.cvtColor(final_foreground, cv2.COLOR_BGR2HSV)
            foreground_mask = (final_foreground[:, :, 0] != 0).astype(np.uint8)
            histograms = HSV.plot_histograms(hsv_image, foreground_mask)


            # 显示直方图在label_3中
            pixmap = QtGui.QPixmap("histograms.png")
            self.label_3.setPixmap(pixmap.scaled(self.label_3.size(), QtCore.Qt.KeepAspectRatio))

            # 判断是否为花色服装
            main_dominant_colors = CutOut.extract_dominant_colors(final_foreground)  # 提取前景的主要颜色
            is_flower = CutOut.is_multicolor(main_dominant_colors, threshold=70) or HSV.find_main_color(
                histograms, threshold=0.500)

            if is_flower:  # 如果服装是花色
                CutOut.display_colors(main_dominant_colors, title="Flower Clothing: Top 2 Colors")  # 显示花色服装的主要颜色
            else:  # 如果服装是纯色
                CutOut.display_colors([main_dominant_colors[0]], title="Solid Clothing: Main Color")  # 显示纯色服装的主要颜色

            pixmap = QtGui.QPixmap("main_color.png")
            self.label_4.setPixmap(pixmap.scaled(self.label_4.size(), QtCore.Qt.KeepAspectRatio))





           




    # 修改process_uploaded_image方法以返回final_foreground
    def process_uploaded_image(self, image_path):
        image = cv2.imread(image_path)
        if image is not None:
            cropped_image = image  # 在这里我们使用整个图像作为裁剪后的图像

            complexity = CutOut.determine_background_complexity(cropped_image)
            if (complexity == "simple"):
                rect = CutOut.generate_initial_rect_simple(cropped_image)
            else:
                rect = CutOut.generate_initial_rect_complex(cropped_image)

            debug_image = cropped_image.copy()
            cv2.rectangle(debug_image, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)

            result, s_image = CutOut.apply_grabcut(cropped_image, rect)

            x_image = CutOut.extract_background_from_s(s_image)
            final_foreground = CutOut.extract_foreground_from_x(cropped_image, x_image)

            # 返回最终的前景图像
            return final_foreground
        else:
            print("图片读取失败，请检查图片路径或文件格式。")
            return None


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

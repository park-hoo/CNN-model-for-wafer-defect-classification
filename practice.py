import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QVBoxLayout, QWidget, QScrollArea
from PyQt5.QtGui import QPixmap, QImage, QImageReader
from PyQt5.QtCore import Qt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PIL import Image
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QSizePolicy
from PyQt5.QtWidgets import QTextEdit, QDockWidget, QPlainTextEdit, QScrollBar

# 이미지 데이터 증강을 위한 ImageDataGenerator 설정
image_gen = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rescale=1/255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# 이미지 데이터셋 경로 설정
training_image = image_gen.flow_from_directory(
    "wafer/dataset/class",
    target_size=(100, 100),
    batch_size=25,
    color_mode='grayscale'
)

# 모델 경로 설정
model_path = 'gray_image_class3.h5'

# PyQt 애플리케이션 클래스
class ImageClassificationApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # 모델 불러오기
        self.model = load_model(model_path)

        # UI 초기화
        self.initUI()

    def initUI(self):
        # UI 요소 초기화
        self.setWindowTitle("Image Classification App")
        self.setGeometry(100, 100, 1600, 900)

        # 이미지 출력을 위한 라벨
        self.image_label = QLabel(self)
        self.image_label.setGeometry(50, 50, 300, 300)

        # 분류 결과를 출력하기 위한 라벨
        self.result_label = QLabel(self)
        self.result_label.setGeometry(50, 360, 650, 30)
        self.result_label.setAlignment(Qt.AlignCenter)

        # 그래프 크기 조절을 위한 QSizePolicy 설정
        size_policy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        size_policy.setVerticalStretch(1)  # 수직으로 늘어나도록 설정

        # 그래프 위젯 초기화
        self.graph_widget = QScrollArea(self)
        self.graph_widget.setGeometry(460, 50, 900, 800)
        self.graph_layout = QVBoxLayout(self.graph_widget)
        self.graph_canvas = FigureCanvas(Figure(figsize=(8, 6)))  # 그래프 크기 조절
        self.graph_layout.addWidget(self.graph_canvas)
        self.graph_canvas.setSizePolicy(size_policy)  # 크기 조절 속성 설정
        self.graph_widget.setWidgetResizable(True)
        
        # 파일 선택 버튼
        self.select_button = QPushButton("Select Image", self)
        self.select_button.setGeometry(50, 50, 200, 40)
        self.select_button.clicked.connect(self.loadImage)

        # 분류 버튼
        self.classify_button = QPushButton("Classify", self)
        self.classify_button.setGeometry(50, 300, 200, 40)
        self.classify_button.clicked.connect(self.processImage)

        # 변수 초기화
        self.image_path = None
        self.classification_result = None
        
                # 로그 창 초기화
        self.log_widget = QScrollArea(self)
        self.log_widget.setGeometry(50, 400, 350, 400)
        self.log_widget.setWidgetResizable(True)
        self.log_text = QPlainTextEdit(self.log_widget)
        self.log_text.setReadOnly(True)
        self.log_widget.setWidget(self.log_text)

        # 변수 초기화
        self.image_path = None
        self.classification_result = None

    def loadImage(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image File", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.gif)", options=options
        )

        if file_path:
            self.image_path = file_path
            self.classification_result = None
            # 이미지를 QLabel에 표시
            image_data = QImage(self.image_path)
            pixmap = QPixmap.fromImage(image_data)
            self.image_label.setPixmap(pixmap)
            self.image_label.setAlignment(Qt.AlignCenter)
            self.result_label.clear()

            # self.graph_canvas.setFixedSize(pixmap.width(), pixmap.height())

    def processImage(self):
        if self.image_path:
            img = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
            img = np.expand_dims(img, axis=-1)  # 이미지 데이터에 채널 차원 추가

            test1_proc = image.load_img(self.image_path, color_mode='grayscale', target_size=(100, 100))
            test1_proc_1 = image.img_to_array(test1_proc)
            t1 = np.expand_dims(test1_proc_1, axis=0)
            t1 = t1 / 255
            pred_classes = self.model.predict(t1)
            pred = np.argmax(pred_classes, axis=1)

            original_label = os.path.basename(os.path.dirname(self.image_path))
            self.log_text.appendPlainText(f"Image Path: {self.image_path}")
            self.log_text.appendPlainText(f"Processing Image: {self.image_path}")
            # 그래프 그리기
            fig, axes = plt.subplots(2, 1, figsize=(8, 6))
            # axes[0].set_box_aspect(1.5)  # 종횡비 설정

            # 위쪽 서브플롯: 원본 이미지 표시
            pic = Image.open(self.image_path)
            axes[0].set_title("Original Image: " + original_label)
            axes[0].imshow(pic)
            axes[0].axis('off')
            
            # 아래쪽 서브플롯: 예측 결과 막대 그래프
            s1 = pd.Series(pred_classes.ravel(), index=training_image.class_indices.keys())
            s1.plot(kind='barh', ax=axes[1], color='green')
            axes[1].set_xlabel("Prediction")
            axes[1].set_title("Prediction Bar Chart")
            
            self.log_text.appendPlainText("Image processing completed")
            # 서브플롯 간 간격 조정
            plt.subplots_adjust(hspace=0.5)
            plt.tight_layout()

            # # 결과 라벨 설정
            # if self.classification_result is not None:
            #     self.result_label.setText(f"Prediction: {self.classification_result}")

            # 그래프를 그린 후 PyQt 화면에 표시
            # self.graph_layout.addWidget(self.result_label)
            self.graph_layout.addWidget(FigureCanvas(fig))
            self.result_label.clear()
            

# def main():
#     app = QApplication(sys.argv)
#     window = ImageClassificationApp()
#     window.show()
#     sys.exit(app.exec_())

# if __name__ == '__main__':
#     main()

def main():
    app = QApplication(sys.argv)
    window = ImageClassificationApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

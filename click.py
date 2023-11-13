
import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QVBoxLayout, QWidget, QScrollArea, QTableWidget, QTableWidgetItem
from PyQt5.QtGui import QPixmap, QImage, QImageReader,QIcon
from PyQt5.QtCore import Qt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PIL import Image
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QSizePolicy, QPlainTextEdit,QHeaderView

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

        # 그래프를 초기화할 변수
        self.graph_canvas = None

        # UI 초기화
        self.initUI()

    def initUI(self):
        # UI 요소 초기화
        
        #Window 창 설정
        self.setWindowIcon(QIcon("ap1.png")) 
        self.setWindowTitle("Wafer Defect Classification")
        self.setGeometry(100, 100, 1400, 900)
        self.setStyleSheet("background-color: lightgray;") 

        # 이미지 출력을 위한 라벨
        self.image_label = QLabel(self)
        self.image_label.setGeometry(50, 50, 300, 300)

        # 분류 결과를 출력하기 위한 라벨
        self.result_label = QLabel(self)
        self.result_label.setGeometry(50, 360, 650, 30)
        self.result_label.setAlignment(Qt.AlignCenter)

        # 그래프 크기 조절을 위한 QSizePolicy 설정
        # size_policy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # size_policy.setVerticalStretch(1)  # 수직으로 늘어나도록 설정

        # 그래프 위젯 초기화
        self.graph_widget = QScrollArea(self)
        self.graph_widget.setGeometry(460, 50, 900, 600)
        self.graph_layout = QVBoxLayout(self.graph_widget)
        self.graph_widget.setWidgetResizable(True)

        # 파일 선택 버튼
        self.select_button = QPushButton("Select Image", self)
        self.select_button.setGeometry(50, 200, 200, 40)
        self.select_button.clicked.connect(self.loadImage)

        # 분류 버튼
        self.classify_button = QPushButton("Classification", self)
        self.classify_button.setGeometry(50, 370, 200, 40)
        self.classify_button.clicked.connect(self.processImage)

        # 폴더 선택 버튼
        self.select_folder_button = QPushButton("Select Folder", self)
        self.select_folder_button.setGeometry(250, 200, 200, 40)
        self.select_folder_button.clicked.connect(self.processImagesF)

        # 표 위젯
        self.table_widget = QTableWidget(self)
        self.table_widget.setGeometry(460, 690, 900, 150)

        # 표 헤더 설정
        table_header = ["No", "Original Label","Original_filename","Predicted Label", "Accuracy", "Result"]
        self.table_widget.setColumnCount(len(table_header))
        self.table_widget.setHorizontalHeaderLabels(table_header)
        
        # 연결: 표 행 클릭 시 그래프 업데이트
        self.table_widget.cellClicked.connect(self.updateGraph)  # 표 행 클릭 이벤트 연결
        self.table_widget.setSelectionBehavior(QTableWidget.SelectRows)  # 행 전체 선택
        self.table_widget.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table_widget.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table_widget.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table_widget.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.table_widget.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.table_widget.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)

        # 로그 창 초기화
        self.log_widget = QScrollArea(self)
        self.log_widget.setGeometry(50, 445, 350, 400)
        self.log_widget.setWidgetResizable(True)
        self.log_text = QPlainTextEdit(self.log_widget)
        self.log_text.setReadOnly(True)
        self.log_widget.setWidget(self.log_text)

        # 변수 초기화
        self.image_path = None
        self.classification_result = None
        self.row_counter = 0  # 표에 추가된 행 수를 추적
        
        
        #AP시스템 로고 QPixmap
        image_path="ap.png"
    
    
        # 이미지 파일을 QPixmap 객체로
        logo = QPixmap(image_path)
        self.image_display = QLabel(self)
        
        # QLabel에 QPixmap을 설정하여 이미지 표시
        self.image_display.setPixmap(logo)
        self.image_display.setPixmap(logo)
        self.image_display.setScaledContents(True)
        self.image_display.setGeometry(50, 50, 400, 130)  # 이미지 크기 및 위치 조절


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
            # self.image_label.setAlignment(Qt.AlignCenter)
            self.image_label.setGeometry(100, 155, 300, 300)
            self.result_label.clear()
            self.log_text.appendPlainText(f"\nImage Path: {self.image_path}")

    
    #단일 이미지 프로세싱
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
            original_filename = os.path.basename(self.image_path)

            # Fig=frame, axes는 plot
            fig, axes = plt.subplots(2, 1, figsize=(8, 6))

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
            plt.subplots_adjust(hspace=0.5)
            plt.tight_layout()
            self.log_text.appendPlainText(f"\nProcessing Image: {self.image_path}")
            self.log_text.appendPlainText("\n\nImage processing completed.\n\n")

            # 그래프 Canvas 업데이트
            if self.graph_canvas is None:
                self.graph_canvas = FigureCanvas(fig)
                self.graph_layout.addWidget(self.graph_canvas)
            else:
                self.graph_canvas.figure = fig
                self.graph_canvas.draw()

            self.result_label.clear()
            predicted_label = list(training_image.class_indices.keys())[pred[0]]
            if original_label == predicted_label:
                result = "Match"
            else:
                result = "Mismatch"
            accuracy = pred_classes[0][pred[0]]
            self.log_text.appendPlainText("----------------------------------------------")
            self.log_text.appendPlainText(f"Original Image: {original_label} \n\nPredicted Image: {predicted_label} \n\nResult: {result} \n\nAccuracy: {accuracy*100:0.2f}%")
            self.log_text.appendPlainText("----------------------------------------------")
            
            # 표에 결과 추가
            self.row_counter += 1
            self.table_widget.setRowCount(self.row_counter)
            self.table_widget.setItem(self.row_counter - 1, 0, QTableWidgetItem(str(self.row_counter)))
            self.table_widget.setItem(self.row_counter - 1, 1, QTableWidgetItem(original_label))
            self.table_widget.setItem(self.row_counter - 1, 2, QTableWidgetItem(original_filename))
            self.table_widget.setItem(self.row_counter - 1, 3, QTableWidgetItem(predicted_label))
            self.table_widget.setItem(self.row_counter - 1, 4, QTableWidgetItem(f"{accuracy*100:0.2f}%"))
            self.table_widget.setItem(self.row_counter - 1, 5, QTableWidgetItem(result))

            self.classification_result = predicted_label
    
    
    #폴더 전체 프로세싱 (F=Folder)
    def processImagesF(self):
        folder = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        if folder:
            image_files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
            total_images = len(image_files)
            self.row_counter = 0
            match_count = 0

            self.log_text.appendPlainText(f"\nProcessing {total_images} images from folder: {folder}")

            for image_file in image_files:
                image_path = os.path.join(folder, image_file)
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                img = np.expand_dims(img, axis=-1)

                test_proc = image.load_img(image_path, color_mode='grayscale', target_size=(100, 100))
                test_proc = image.img_to_array(test_proc)
                test_proc = np.expand_dims(test_proc, axis=0)
                test_proc = test_proc / 255

                pred_classes = self.model.predict(test_proc)
                pred = np.argmax(pred_classes, axis=1)

                original_label = os.path.basename(os.path.dirname(image_path))
                original_filename = os.path.basename(image_path)

                # 결과 표에 결과 추가
                accuracy = pred_classes[0][pred[0]]
                predicted_label = list(training_image.class_indices.keys())[pred[0]]
                result = "Match" if original_label == predicted_label else "Mismatch"
                self.row_counter += 1
                self.table_widget.setRowCount(self.row_counter)
                self.table_widget.setItem(self.row_counter - 1, 0, QTableWidgetItem(str(self.row_counter)))
                self.table_widget.setItem(self.row_counter - 1, 1, QTableWidgetItem(original_label))
                self.table_widget.setItem(self.row_counter - 1, 2, QTableWidgetItem(original_filename))
                self.table_widget.setItem(self.row_counter - 1, 3, QTableWidgetItem(predicted_label))
                self.table_widget.setItem(self.row_counter - 1, 4, QTableWidgetItem(f"{accuracy*100:0.2f}%"))
                self.table_widget.setItem(self.row_counter - 1, 5, QTableWidgetItem(result))

                if result == "Match":
                    match_count += 1

            accuracy_percentage = (match_count / total_images) * 100
            self.log_text.appendPlainText(f"\nImage processing completed.")
            self.log_text.appendPlainText("\n----------------------------------------------")
            self.log_text.appendPlainText(f"Processed {total_images} images.")
            self.log_text.appendPlainText(f"\nMatched: {match_count} images out of {total_images} (Accuracy: {accuracy_percentage:.2f}%)")
            self.log_text.appendPlainText("----------------------------------------------")

            self.classification_result = None


    #표 클릭 후 결과값 출력
    def updateGraph(self, row):
        if row < self.row_counter:
            # 예측 결과를 가져오기
            original_label_item = self.table_widget.item(row, 1)
            original_filename_item = self.table_widget.item(row, 2)
            predicted_label_item = self.table_widget.item(row, 3)
            accuracy_item = self.table_widget.item(row, 4)
            result_item = self.table_widget.item(row, 5)

            img = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
            img = np.expand_dims(img, axis=-1)  # 이미지 데이터에 채널 차원 추가


            original_label = original_label_item.text()
            predicted_label = predicted_label_item.text()
            accuracy = float(accuracy_item.text().replace('%', ''))
            result = result_item.text()
            
            
            fig, axes = plt.subplots(2, 1, figsize=(8, 6))

            # 위쪽 서브플롯: 원본 이미지 표시
            image_path = os.path.join("wafer/dataset/consistency", original_label, original_filename_item.text())
            image_path=image_path.replace("\\","/")
            pic = Image.open(image_path)
            
            axes[0].set_title("Original Image: " + original_label)
            axes[0].imshow(pic)
            axes[0].axis('off')

            # 아래쪽 서브플롯: 예측 결과 막대 그래프
            pic1 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            pic1 = np.expand_dims(pic1, axis=-1)

            test1_proc = image.load_img(image_path, color_mode='grayscale', target_size=(100, 100))
            test1_proc_1 = image.img_to_array(test1_proc)
            t1 = np.expand_dims(test1_proc_1, axis=0)
            t1 = t1 / 255

            pred_classes = self.model.predict(t1)
            pred = np.argmax(pred_classes, axis=1)

            s1 = pd.Series(pred_classes.ravel(), index=training_image.class_indices.keys())
            s1.plot(kind='barh', ax=axes[1], color='green')
            axes[1].set_xlabel("Prediction")
            axes[1].set_title("Prediction Bar Chart")
            plt.subplots_adjust(hspace=0.2)
            plt.tight_layout()
            
            
            #그래프 작아지는 현상 해결 완료
            if self.graph_canvas is not None:
                self.graph_layout.removeWidget(self.graph_canvas)
                self.graph_canvas.deleteLater()
                self.graph_canvas = None
                
            self.graph_canvas = FigureCanvas(fig)
            self.graph_layout.addWidget(self.graph_canvas)

            self.result_label.clear()
        else:
            self.result_label.setText("Select a valid row.")
            
            
        #     # 그래프 업데이트
        #     if self.graph_canvas is None:
        #         self.graph_canvas = FigureCanvas(fig)
        #         self.graph_layout.addWidget(self.graph_canvas)
        #     else:
        #         self.graph_canvas.figure = fig
        #         self.graph_canvas.draw()

        #     self.result_label.clear()
        # else:
        #     self.result_label.setText("Select a valid row.")


def main():
    # app = QApplication(sys.argv)
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = ImageClassificationApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()



import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtCore import Qt
import tensorflow as tf
import os  # To check for file existence
from tensorflow.keras.models import load_model


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set window parameters----------
        self.HEIGHT = 600
        self.WIDTH = 800
        self.setWindowTitle("Image Classifier")
        self.setGeometry(100, 100, self.WIDTH, self.HEIGHT)

        # Main window background color and font---------
        self.setStyleSheet("background-color: #f0f0f0; font-family: Arial;")

        # Video label-----------
        self.video_label = QLabel(self)
        self.video_label.setGeometry(20, 20, 400, 400)
        self.video_label.setStyleSheet("border: 2px solid black; background-color: white;")

        # Buttons--------------
        # Button to browse and select image file
        self.browse_button = QPushButton('Browse Image', self)
        self.browse_button.setGeometry(80, 480, 150, 60)
        self.browse_button.setStyleSheet("background-color: #007bff; color: white; border: none; padding: 5px;")
        self.browse_button.clicked.connect(self.browse_image)

        # Classify button
        self.classify_button = QPushButton("Classify Image", self)
        self.classify_button.setGeometry(280, 480, 150, 60)
        self.classify_button.setStyleSheet("background-color: #28a745; color: white; border: none; padding: 5px;")
        self.classify_button.clicked.connect(self.classify_image)

        # Label to display classification result
        self.result_label = QLabel(self)
        self.result_label.setGeometry(490, 480, 250, 60)
        self.result_label.setStyleSheet("background-color: #ffffff; border: 2px solid black; padding: 5px;")
        self.result_label.setAlignment(Qt.AlignCenter)

        # Load the model
        self.model = load_model('my_model2_v12.keras')
        self.classes = ["glioma", "meningioma", "pituitary_tumor"]

        # Set up a timer to update the frame regularly
        self.timer = self.startTimer(500)  # Update every 500 ms

    def timerEvent(self, event):
        # Check if 'captured_frame.jpg' exists
        if os.path.exists('captured_frame.jpg'):
            # Load the image
            frame = cv2.imread('captured_frame.jpg')

            # Convert to RGB (since OpenCV loads it in BGR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert to QImage and scale it
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            convert_to_qt_format = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            p = convert_to_qt_format.scaled(400, 400, Qt.KeepAspectRatio)

            # Set the pixmap
            self.video_label.setPixmap(QPixmap.fromImage(p))
        else:
            # If no file exists, clear the label (display nothing)
            self.video_label.clear()

    def browse_image(self):
        # Open file dialog to select an image file
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Images (*.png *.jpg)")
        file_dialog.setViewMode(QFileDialog.Detail)
        if file_dialog.exec_():
            file_names = file_dialog.selectedFiles()
            if file_names:
                file_path = file_names[0]
                image = cv2.imread(file_path)
                cv2.imwrite('captured_frame.jpg', image)

    def apply_clahe(self, image_path):
        # Read the MRI image
        image = cv2.imread(image_path)

        if len(image.shape) == 2:  # Grayscale image
            clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
            enhanced_image = clahe.apply(image)
        else:  # Color image
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)  # Convert to LAB color space
            l, a, b = cv2.split(lab)  # Split into L, A, B channels
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_clahe = clahe.apply(l)  # Apply CLAHE to the L channel
            lab_clahe = cv2.merge((l_clahe, a, b))  # Merge the CLAHE-enhanced L channel with A and B
            enhanced_image = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)  # Convert back to BGR

        return enhanced_image

    def classify_image(self):
        image_path = 'captured_frame.jpg'
        enhanced_image = self.apply_clahe(image_path)

        # Resize the image to (64, 64)
        resized = cv2.resize(enhanced_image, (64, 64))

        # Normalize pixel values to [0, 1] and expand dimensions for the model
        processed = np.expand_dims(resized, axis=-1)  # Add channel dimension
        processed = np.expand_dims(processed, axis=0)  # Add batch dimension
        processed = processed / 255.0  # Normalize

        prediction = self.model.predict(processed)
        class_index = np.argmax(prediction)
        confidence = prediction[0][class_index]
        self.result_label.setText(f'Predicted Label: {self.classes[class_index]}')

if __name__ == "__main__":
    print(tf.__version__)
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


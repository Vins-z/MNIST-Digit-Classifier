import os
import numpy as np
import tensorflow as tf
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
                             QFileDialog, QFrame, QSizePolicy, QComboBox, QInputDialog, QProgressBar, QStatusBar,
                             QGridLayout, QScrollArea, QMessageBox)
from PyQt5.QtGui import QPixmap, QFont, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import time
import models
import utils
from models.simple_ann import create_simple_ann
from models.resnet_like_ann import create_resnet_like_ann
from utils.data_loader import load_data
from utils.image_processor import preprocess_image
from utils.logger import get_logger

logger = get_logger(__name__)

# Model file paths
model_paths = {
    "Simple ANN": os.path.join(os.getcwd(), "mnist_simple_ann_model.h5"),
    "ResNet-like ANN": os.path.join(os.getcwd(), "mnist_resnet_like_ann_model.h5")
}

class TrainingPlotCallback(tf.keras.callbacks.Callback):
    def __init__(self, canvas, training_thread):
        super(TrainingPlotCallback, self).__init__()
        self.canvas = canvas
        self.training_thread = training_thread
        self.epochs = []
        self.train_acc = []
        self.val_acc = []
        self.train_loss = []
        self.val_loss = []

    def on_epoch_end(self, epoch, logs=None):
        self.epochs.append(epoch)
        self.train_acc.append(logs.get('accuracy', 0))
        self.val_acc.append(logs.get('val_accuracy', 0))
        self.train_loss.append(logs.get('loss', 0))
        self.val_loss.append(logs.get('val_loss', 0))
        self.plot_metrics()
        self.training_thread.emit_epoch_status(epoch, logs)

    def plot_metrics(self):
        self.canvas.figure.clear()
        axs = self.canvas.figure.subplots(1, 2)

        axs[0].plot(self.epochs, self.train_acc, label='Train Accuracy')
        axs[0].plot(self.epochs, self.val_acc, label='Validation Accuracy')
        axs[0].set_title("Accuracy")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Accuracy")
        axs[0].legend()

        axs[1].plot(self.epochs, self.train_loss, label='Train Loss')
        axs[1].plot(self.epochs, self.val_loss, label='Validation Loss')
        axs[1].set_title("Loss")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Loss")
        axs[1].legend()

        self.canvas.draw()
        QApplication.processEvents()

class TrainingThread(QThread):
    update_progress = pyqtSignal(int)
    training_finished = pyqtSignal()
    epoch_status = pyqtSignal(int, float, float, float, float)  # Epoch, Train Acc, Val Acc, Train Loss, Val Loss

    def __init__(self, model, callback):
        super().__init__()
        self.model = model
        self.callback = callback

    def run(self):
        history = self.model.fit(
            x_train, y_train,
            validation_data=(x_test, y_test),
            epochs=5,
            callbacks=[self.callback],
            verbose=2
        )
        for i in range(5):
            self.update_progress.emit((i + 1) * 20)
            time.sleep(1)  # Simulate delay
        self.training_finished.emit()

    def emit_epoch_status(self, epoch, logs):
        train_acc = logs.get('accuracy', 0)
        val_acc = logs.get('val_accuracy', 0)
        train_loss = logs.get('loss', 0)
        val_loss = logs.get('val_loss', 0)
        self.epoch_status.emit(epoch, train_acc, val_acc, train_loss, val_loss)

class ImageClassifierApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = None
        self.image_path = None
        self.correct_label = None
        self.current_model_type = "Simple ANN"
        self.training_thread = None  # Initialize training_thread here
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Digit Classification')
        self.setGeometry(100, 100, 1200, 800)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # Left panel for model selection and training
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setStyleSheet("background-color: #e0e0e0; border-radius: 10px; padding: 20px;")

        title = QLabel('Digit Classifier')
        title.setFont(QFont('Arial', 24, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #333333; margin-bottom: 20px;")
        left_layout.addWidget(title)

        model_selection_layout = self.create_model_dropdown()
        left_layout.addLayout(model_selection_layout)

        self.retrain_button = QPushButton('Retrain Model')
        self.retrain_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 10px;
                font-size: 16px;
                border-radius: 5px;
                border: none;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3e8e41;
            }
        """)
        self.retrain_button.clicked.connect(self.retrain_model)
        left_layout.addWidget(self.retrain_button)

        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                width: 10px;
                margin: 0.5px;
            }
        """)
        left_layout.addWidget(self.progress_bar)

        self.canvas = self.create_training_plot()
        left_layout.addWidget(self.canvas)

        main_layout.addWidget(left_panel, 1)

        # Right panel for image upload and classification
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_panel.setStyleSheet("background-color: #e0e0e0; border-radius: 10px; padding: 20px;")

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("""
            background-color: white;
            border: 2px solid #ccc;
            border-radius: 10px;
        """)
        self.image_label.setFixedSize(280, 280)
        right_layout.addWidget(self.image_label)

        upload_button = QPushButton('Upload Image')
        upload_button.setStyleSheet("""
            QPushButton {
                background-color: #008CBA;
                color: white;
                padding: 10px;
                font-size: 16px;
                border-radius: 5px;
                border: none;
            }
            QPushButton:hover {
                background-color: #007B9A;
            }
            QPushButton:pressed {
                background-color: #006B8A;
            }
        """)
        upload_button.clicked.connect(self.upload_image)
        right_layout.addWidget(upload_button)

        self.classify_button = QPushButton('Classify Image')
        self.classify_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                padding: 10px;
                font-size: 16px;
                border-radius: 5px;
                border: none;
            }
            QPushButton:hover {
                background-color: #E53935;
            }
            QPushButton:pressed {
                background-color: #D32F2F;
            }
            QPushButton:disabled {
                background-color: #BDBDBD;
                color: #757575;
            }
        """)
        self.classify_button.clicked.connect(self.classify_image)
        self.classify_button.setEnabled(False)
        right_layout.addWidget(self.classify_button)

        self.result_label = QLabel('')
        self.result_label.setFont(QFont('Arial', 16))
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("background-color: white; padding: 10px; border-radius: 5px;")
        right_layout.addWidget(self.result_label)

        self.correct_button = QPushButton('Correct Prediction')
        self.correct_button.setStyleSheet("""
            QPushButton {
                background-color: #555555;
                color: white;
                padding: 10px;
                font-size: 16px;
                border-radius: 5px;
                border: none;
            }
            QPushButton:hover {
                background-color: #444444;
            }
            QPushButton:pressed {
                background-color: #333333;
            }
            QPushButton:disabled {
                background-color: #BDBDBD;
                color: #757575;
            }
        """)
        self.correct_button.clicked.connect(self.correct_prediction)
        self.correct_button.setEnabled(False)
        right_layout.addWidget(self.correct_button)

        main_layout.addWidget(right_panel, 1)

        # Status bar
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet("background-color: #333333; color: white; font-size: 14px;")
        self.setStatusBar(self.status_bar)

        self.initialize_model()

    def create_model_dropdown(self):
        model_selection_layout = QHBoxLayout()
        model_label = QLabel("Select Model:")
        model_label.setFont(QFont('Arial', 14))
        model_label.setStyleSheet("color: black")
        self.model_dropdown = QComboBox()
        self.model_dropdown.setFont(QFont('Arial', 14))
        self.model_dropdown.addItems(["Simple ANN", "ResNet-like ANN"])
        self.model_dropdown.setStyleSheet("""
            QComboBox {
                padding: 5px;
                border: 1px solid #ccc;
                border-radius: 3px;
                background-color: white;
                color: black;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 15px;
                border-left-width: 1px;
                border-left-color: darkgray;
                border-left-style: solid;
                border-top-right-radius: 3px;
                border-bottom-right-radius: 3px;
            }
            QComboBox::down-arrow {
                image: url(down_arrow.png);
            }
        """)
        self.model_dropdown.currentTextChanged.connect(self.change_model)
        model_selection_layout.addWidget(model_label)
        model_selection_layout.addWidget(self.model_dropdown)
        return model_selection_layout

    def create_training_plot(self):
        fig, _ = plt.subplots(1, 2, figsize=(10, 4))
        canvas = FigureCanvas(fig)
        return canvas

    def change_model(self, model_type):
        self.current_model_type = model_type
        self.initialize_model()

    def initialize_model(self):
        model_path = model_paths[self.current_model_type]
        if os.path.exists(model_path):
            self.load_model()
        else:
            self.train_model()

    def load_model(self):
        model_path = model_paths[self.current_model_type]
        self.status_bar.showMessage(f"Loading {self.current_model_type} model...")
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            self.status_bar.showMessage(f"Loaded saved {self.current_model_type} model!")
        else:
            self.status_bar.showMessage(f"No saved {self.current_model_type} model found. Training new model...")
            self.train_model()

    def save_model(self):
        model_path = model_paths[self.current_model_type]
        self.model.save(model_path)
        if os.path.exists(model_path):
            self.status_bar.showMessage(f'{self.current_model_type} model trained and saved!')
        else:
            self.status_bar.showMessage(f'Failed to save {self.current_model_type} model!')

    def train_model(self):
        if self.current_model_type == "Simple ANN":
            self.model = create_simple_ann()
        else:  # ResNet-like ANN
            self.model = create_resnet_like_ann()

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Create the callback with the canvas and training_thread
        self.training_thread = TrainingThread(self.model, None)  # Initialize with None
        callback = TrainingPlotCallback(self.canvas, self.training_thread)
        self.training_thread.callback = callback  # Set the callback after initialization

        self.training_thread.update_progress.connect(self.update_progress_bar)
        self.training_thread.training_finished.connect(self.on_training_finished)

        self.status_bar.showMessage(f"Training {self.current_model_type} model...")
        self.retrain_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.training_thread.start()

    def update_progress_bar(self, value):
        self.progress_bar.setValue(value)

    def on_training_finished(self):
        self.save_model()
        self.retrain_button.setEnabled(True)

    def retrain_model(self):
        self.status_bar.showMessage(f'Retraining {self.current_model_type} model...')
        self.train_model()

    def upload_image(self):
        file_dialog = QFileDialog()
        self.image_path, _ = file_dialog.getOpenFileName(self, 'Open Image', '', 'Image Files (*.jpeg *.jpg *.png)')
        if self.image_path:
            try:
                pixmap, preprocessed_image = preprocess_image(self.image_path)
                self.image_label.setPixmap(pixmap)
                self.classify_button.setEnabled(True)
                self.status_bar.showMessage('Image uploaded and preprocessed successfully!')

                # Store the preprocessed image for classification
                self.preprocessed_image = preprocessed_image
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load image: {str(e)}")
                self.status_bar.showMessage('Failed to upload image.')
        else:
            self.status_bar.showMessage('No image selected.')

    def classify_image(self):
        if hasattr(self, 'preprocessed_image') and self.model:
            try:
                # Normalize the image
                img_array = self.preprocessed_image.astype('float32') / 255.0

                # Reshape for model input
                img_array = img_array.reshape((1, 28, 28, 1))

                prediction = self.model.predict(img_array)
                predicted_class = np.argmax(prediction)
                confidence = prediction[0][predicted_class] * 100

                self.result_label.setText(f'Predicted Digit: {predicted_class}\nConfidence: {confidence:.2f}%')
                self.correct_button.setEnabled(True)
                self.status_bar.showMessage('Image classified successfully!')
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to classify image: {str(e)}")
                self.status_bar.showMessage('Failed to classify image.')
        else:
            self.status_bar.showMessage('Please upload an image and ensure a model is loaded.')

    def correct_prediction(self):
        correct_digit, ok = QInputDialog.getInt(self, 'Correct Prediction', 'Enter the correct digit:', min=0, max=9)
        if ok:
            self.correct_label = correct_digit
            self.status_bar.showMessage(f'Correct label set to {correct_digit}')
            # Here you could implement logic to use this correction for further training or evaluation

if __name__ == '__main__':
    app = QApplication([])
    ex = ImageClassifierApp()
    ex.show()
    app.exec_()

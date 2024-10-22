from PIL import Image
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
import numpy as np

def preprocess_image(image_path):
    try:
        with Image.open(image_path) as pil_image:
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            gray_img = pil_image.convert('L')
            resized_img = gray_img.resize((28, 28), Image.LANCZOS)
            qimg = QImage(resized_img.tobytes(), 28, 28, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qimg).scaled(280, 280, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            preprocessed_image = np.array(resized_img)
            return pixmap, preprocessed_image
    except Exception as e:
        raise e

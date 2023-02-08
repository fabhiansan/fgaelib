import cv2
import matplotlib.pyplot as plt
import numpy as np

HAARCASCADE_MODEL = "./model/haarcascade_frontalface_default.xml"

class FaceDetector():
    def __init__(self, model_dir:str = HAARCASCADE_MODEL) -> None:
        """
        Face Detector Class.

        :param model_dir: Directory of Haarcascade Frontal Face Classifier
        """
        self.model_dir = model_dir
        self.face_detector = cv2.CascadeClassifier(self.model_dir)

    def predict_face(self, img: np.ndarray, padding:int = 5) -> list:
        """
        Predict the face in the image given

        :param padding: pixels to extend the face image to get other features
        """
        if img.shape[2] > 1:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        faces = self.face_detector.detectMultiScale(img, 1.1, 4)
        self.n_faces = len(faces)
        self.faces_in_image = []
        for (x, y, w, h) in faces:
            face_cropped= img[y-padding:y+h+padding, x-padding:x+w+padding]
            self.faces_in_image.append(face_cropped)
        return self.faces_in_image

    def show_faces(self):
        plt.figure(figsize=(12, 9))
        if len(self.faces_in_image) > 1:
            for i in range(len(self.faces_in_image)):
                plt.subplot(1, len(self.faces_in_image), i+1)
                plt.imshow(self.faces_in_image[i])
                plt.axis("off")
        else:
            plt.imshow(self.faces_in_image[0])
            plt.axis("off")

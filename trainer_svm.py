import os

import cv2
import kornia
import numpy as np

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


def fit_one_epoch(train_gen, val_gen):
    loss = 0
    train_set = set()
    print('Start Train')

    model = SVC(kernel='rbf', C=10.0, gamma='scale')

    for iteration, batch in enumerate(train_gen):
        if iteration >= 1:#100
            break

        train_images, train_label = batch[0], batch[1]  # image (B,C,H,W)   label (B)
        print(np.shape(train_images), np.shape(train_label))
        model.fit(train_images, train_label)

    print('Start test')

    acc = 0
    for iteration, batch in enumerate(val_gen):
        if iteration >= 1:
            break

        val_images, val_label = batch[0], batch[1]
        val_pred  = model.predict(val_images)

        accuracy = accuracy_score(val_label, val_pred)
        report = classification_report(val_label, val_pred,zero_division=1)
        print("Accuracy:", accuracy)
        print("Classification report:", report)


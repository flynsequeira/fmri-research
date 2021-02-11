import numpy as np
from nibabel.testing import data_path
import nibabel as nib
import os
import cv2
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt


class Model:
    # https: // www.tensorflow.org / tutorials / images / cnn
    def __init__(self, train_x_data, train_y_data, test_x_data, test_y_data, test_patients_numbers):
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', input_shape=(199, 190, 1)))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(2))
        # self.model.add(layers.Activation('softmax'))
        self.train_x = train_x_data
        self.train_y = train_y_data
        self.test_x = test_x_data
        self.test_y = test_y_data
        self.test_patients_numbers = test_patients_numbers
        self.checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='training-checkpoint.ckpt',
                                                             save_weights_only=True,
                                                             verbose=1)

    def use_pretrained_model(self):
        self.model.load_weights('training-checkpoint.ckpt')

    def train(self, optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)):
        self.model.compile(optimizer=optimizer,
                           loss=loss,
                           metrics=['accuracy'])
        history = self.model.fit(self.train_x, self.train_y, epochs=15,
                                 validation_data=(self.test_x, self.test_y),
                                 callbacks=[self.checkpoint])

        return history

    def test(self, unhealthy_threshold: int = 70):
        current_number = 0
        predictions = []
        patient_slides_prediction = []
        for slide_index, slide in enumerate(self.test_x):
            if self.test_patients_numbers[slide_index] != current_number:
                current_number += 1
                if patient_slides_prediction.count(0) >= unhealthy_threshold:
                    predictions.append(patient_slides_prediction.count(1)/len(patient_slides_prediction))
                else:
                    predictions.append(patient_slides_prediction.count(1)/len(patient_slides_prediction))
                patient_slides_prediction = []
            pred = self.model.predict(np.array([slide]))
            patient_slides_prediction.append(np.argmax(pred[0]))
        return predictions

    def accuracy(self, predictions):
        # print(len(prediction))
        print(predictions)
        # return sum(np.array(predictions) == np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))/20

    def accuracy_graph(self, history_data):
        # plt.plot(history.history_data['accuracy'], label='accuracy')
        # plt.plot(history.history_data['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.5, 1])
        plt.legend(loc='lower right')


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


# Training or Testing
def get_data(data_category='Training'):
    all_slides = []
    labels = []
    patient_numbers = []
    current_counter = 0
    for label in ['patient', 'health']:
        for i in range(1, 11):
            try:
                data_registered_space = nib.load('./' + data_category + '/' + label +
                                                 '/sub' + str(i) + '/T1_bet_2_0413.nii.gz')
            except:
                if label == 'patient':
                    continue
                else:
                    return np.array(all_slides), np.array(labels), patient_numbers
            whole_brain = data_registered_space.get_fdata()
            # Reshape 2D into 3D where axis-2 is the rgb value, where you use greyscale instead of rgb
            # eg. Greyscale value - 0.5 is represented as [0.5]
            patients_slides = [np.expand_dims(slide, axis=2) for slide in whole_brain]
            if label == 'patient':
                for _ in whole_brain:
                    patient_numbers.append(current_counter)
                    labels.append(0)
            else:
                for _ in whole_brain:
                    patient_numbers.append(current_counter)
                    labels.append(1)
            all_slides.extend(patients_slides)
            current_counter += 1
    return np.array(all_slides), np.array(labels), patient_numbers


train_x, train_y, train_patient_numbers = get_data('Training')
test_x, test_y, test_patient_numbers = get_data('Testing')
print(set(train_patient_numbers))
print(set(test_patient_numbers))
cnn_model = Model(train_x, train_y, test_x, test_y, test_patient_numbers)
# # print('pretraining')
# cnn_model.use_pretrained_model()
history = cnn_model.train()
# print('training done. Test:')
# prediction = cnn_model.test(5)
# print('predictions: ', prediction)
# print('Accuracy:')
# print(cnn_model.accuracy(prediction))

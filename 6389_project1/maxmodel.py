import nibabel as nib

import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
import numpy as np

trainx = [(nib.load('./Training/health/sub'+str(i)+'/T1_bet_2_0413.nii.gz').get_fdata()) for i in range(1,11)]
trainx2 = [nib.load('./Training/patient/sub'+str(i)+'/T1_bet_2_0413.nii.gz').get_fdata() for i in range(1,11)]
trainx.extend(trainx2)
trainx = np.array(trainx)
trainy = [0 for i in range(0,10)]
trainy.extend([1 for i in range(0,10)])
testx = [nib.load('./Testing/health/sub'+str(i)+'/T1_bet_2_0413.nii.gz').get_fdata() for i in range(1,6)]
testx2 = [nib.load('./Testing/patient/sub'+str(i)+'/T1_bet_2_0413.nii.gz').get_fdata() for i in range(1,6)]
testx.extend(testx2)
testx = np.array(testx)
testy = [0 for i in range(0,5)]
testy.extend([1 for i in range(0,5)])
trainx = np.expand_dims(trainx, axis = 4)
testx = np.expand_dims(testx, axis = 4)

# model
# Create the model
model = Sequential()
model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=trainx[0].shape))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(BatchNormalization(center=True, scale=True))
model.add(Dropout(0.5))
model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform'))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(BatchNormalization(center=True, scale=True))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(lr=0.001),
              metrics=['accuracy'])
model.summary()
# Fit data to model
history = model.fit(trainx, to_categorical(trainy),
            batch_size=3,
            epochs=10,
            verbose=1,
            validation_split=0.3)
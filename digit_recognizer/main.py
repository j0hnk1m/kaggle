import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, LeakyReLU
from keras.optimizers import RMSprop
from keras.utils.np_utils import to_categorical
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Prepare training data and split it into training/validation sets (cross-validation)
train = pd.read_csv('./train.csv')
x = np.array(train[train.columns[1:]]).astype('float32') / 255.0
x = x.reshape(x.shape[0], 28,  28, 1)
y = to_categorical(np.array(train['label']).astype('int32'), num_classes=10)

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=21)
x_train = (x_train - x_train.mean()) / x_train.std()
x_val = (x_val - x_val.mean()) / x_val.std()

# Build the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), input_shape=(28, 28, 1), activation='relu', padding='same'))
model.add(Conv2D(32, kernel_size=(5, 5), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(128, kernel_size=(4, 4), activation='relu', padding='same'))
model.add(Conv2D(128, kernel_size=(4, 4), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer=RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0), loss='categorical_crossentropy',
              metrics=['accuracy'])
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.00001)

# # Without data augmentation
# model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val), batch_size=128, epochs=64, verbose=2,
#           callbacks=[learning_rate_reduction])

# With data augmentation
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range=0.1,  # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images
datagen.fit(x_train)
model.fit_generator(datagen.flow(x_train, y_train, batch_size=128), epochs=64, validation_data=(x_val, y_val),
                              verbose=2, steps_per_epoch=x_train.shape[0] // 128, callbacks=[learning_rate_reduction])

# Create the submission file based on the testing data
test = pd.read_csv('./test.csv')
x_test = np.array(test).astype('float32') / 255.0
x_test = x_test.reshape(x_test.shape[0], 28,  28, 1)
x_test = (x_test - x_test.mean()) / x_test.std()

predictions = model.predict_classes(x_test, verbose=0)
submissions = pd.DataFrame({'ImageId': list(range(1, len(predictions)+1)), 'Label': predictions})
submissions.to_csv('submission.csv', index=False, header=True)

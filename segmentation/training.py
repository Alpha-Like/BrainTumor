import os
import cv2
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Input #type: ignore
from tensorflow.keras.models import Model #type: ignore
from sklearn.model_selection import train_test_split # type: ignore

# U-Net Model
def unet_model(input_size=(128, 128, 1)):
    inputs = Input(input_size)
    c1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, 3, activation='relu', padding='same')(c1)
    p1 = MaxPooling2D()(c1)
    c2 = Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = Conv2D(128, 3, activation='relu', padding='same')(c2)
    p2 = MaxPooling2D()(c2)
    c3 = Conv2D(256, 3, activation='relu', padding='same')(p2)
    c3 = Conv2D(256, 3, activation='relu', padding='same')(c3)
    p3 = MaxPooling2D()(c3)
    c4 = Conv2D(512, 3, activation='relu', padding='same')(p3)
    c4 = Conv2D(512, 3, activation='relu', padding='same')(c4)
    u1 = UpSampling2D()(c4)
    u1 = Concatenate()([u1, c3])
    c5 = Conv2D(256, 3, activation='relu', padding='same')(u1)
    c5 = Conv2D(256, 3, activation='relu', padding='same')(c5)
    u2 = UpSampling2D()(c5)
    u2 = Concatenate()([u2, c2])
    c6 = Conv2D(128, 3, activation='relu', padding='same')(u2)
    c6 = Conv2D(128, 3, activation='relu', padding='same')(c6)
    u3 = UpSampling2D()(c6)
    u3 = Concatenate()([u3, c1])
    c7 = Conv2D(64, 3, activation='relu', padding='same')(u3)
    c7 = Conv2D(64, 3, activation='relu', padding='same')(c7)
    outputs = Conv2D(1, 1, activation='sigmoid')(c7)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Load dataset
def load_images(image_dir, mask_dir, size=(128, 128)):
    images, masks = [], []
    filenames = os.listdir(image_dir)
    for filename in filenames:
        img = cv2.imread(os.path.join(image_dir, filename), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(os.path.join(mask_dir, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None and mask is not None:
            img = cv2.resize(img, size) / 255.0
            mask = cv2.resize(mask, size) / 255.0
            images.append(img.reshape(size[0], size[1], 1))
            masks.append(mask.reshape(size[0], size[1], 1))
    return np.array(images), np.array(masks), filenames

# Load and split data
image_dir = "dataset/images"
mask_dir = "dataset/masks"
train_images, train_masks, filenames = load_images(image_dir, mask_dir)

train_X, val_X, train_Y, val_Y, train_filenames, val_filenames = train_test_split(train_images, train_masks, filenames, test_size=0.2, random_state=42)

# Train & Save Model
model = unet_model()
model.fit(train_X, train_Y, validation_data=(val_X, val_Y), epochs=20, batch_size=8)
model.save("tumor_segmentation_unet.h5")
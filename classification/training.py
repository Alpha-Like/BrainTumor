import os
import cv2
import numpy as np
import tensorflow as tf # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from sklearn.model_selection import train_test_split # type: ignore

# **Tumor Classes**
TUMOR_CLASSES = {0: "No Tumor", 1: "Meningioma", 2: "Glioma", 3: "Pituitary Tumor"}
NUM_CLASSES = len(TUMOR_CLASSES)

# **Load Dataset**
def load_images_and_labels(image_dir, labels_file, size=(128, 128)):
    images, labels = [], []
    with open(labels_file, 'r') as f:
        label_data = {line.split()[0]: int(line.split()[1]) for line in f.readlines()}  # Read labels

    filenames = os.listdir(image_dir)
    for filename in filenames:
        img = cv2.imread(os.path.join(image_dir, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, size) / 255.0
            images.append(img.reshape(size[0], size[1], 1))
            labels.append(label_data[filename])  # Read label for classification

    return np.array(images), to_categorical(labels, num_classes=NUM_CLASSES)

# **Load & Split Data**
image_dir = "dataset/images"
labels_file = "dataset/labels.txt"  # Format: "filename label"
X, Y = load_images_and_labels(image_dir, labels_file)

train_X, val_X, train_Y, val_Y = train_test_split(X, Y, test_size=0.2, random_state=42)

# **CNN Classification Model**
def build_classifier(input_size=(128, 128, 1), num_classes=NUM_CLASSES):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_size),
        MaxPooling2D(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# **Train & Save Model**
model = build_classifier()
model.fit(train_X, train_Y, validation_data=(val_X, val_Y), epochs=20, batch_size=8)
model.save("tumor_classification.h5")
print("✅ Classification model saved as tumor_classification.h5")

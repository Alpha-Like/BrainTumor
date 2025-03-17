import os
import cv2
import numpy as np
import tensorflow as tf # type: ignore

# Load trained U-Net model
model = tf.keras.models.load_model("tumor_segmentation_unet.h5")

# Function to preprocess image
def preprocess_image(image_path, size=(128, 128)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")
    img_resized = cv2.resize(img, size) / 255.0
    return img, img_resized.reshape(1, size[0], size[1], 1)

# Function to predict and overlay red border
def predict_and_draw(image_path, output_path, size=(128, 128)):
    original_img, img_processed = preprocess_image(image_path, size)
    pred_mask = model.predict(img_processed)[0]
    
    # Convert mask to binary and resize to original image size
    mask = (pred_mask > 0.5).astype(np.uint8) * 255
    mask_resized = cv2.resize(mask, (original_img.shape[1], original_img.shape[0]))

    # Find contours of the tumor mask
    contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Convert grayscale image to BGR
    result_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
    
    # Draw red contours on the original image
    cv2.drawContours(result_img, contours, -1, (0, 0, 255), 2)  # Red border
    
    # Save the result
    cv2.imwrite(output_path, result_img)
    print(f"Marked tumor saved at: {output_path}")

# Take image path as input
image_path = input("Enter the image path: ").strip()
if not os.path.exists(image_path):
    print("Error: Image path does not exist.")
else:
    output_path = os.path.join("predicted_images", "marked_" + os.path.basename(image_path))
    os.makedirs("predicted_images", exist_ok=True)
    predict_and_draw(image_path, output_path)

import os
import cv2
import numpy as np
import tensorflow as tf  # type: ignore

# **Load trained models**
classification_model = tf.keras.models.load_model(os.path.abspath("../tumor_classification.h5"))
segmentation_model = tf.keras.models.load_model(os.path.abspath("../tumor_segmentation_unet.h5"))

# **Tumor Classes**
TUMOR_CLASSES = {0: "No Tumor", 1: "Meningioma", 2: "Glioma", 3: "Pituitary Tumor"}

# **Preprocess image**
def preprocess_image(image_path, size=(128, 128)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")
    img_resized = cv2.resize(img, size) / 255.0
    return img, img_resized.reshape(1, size[0], size[1], 1)

# **Predict tumor type**
def predict_tumor(image_path):
    original_img, img_processed = preprocess_image(image_path)
    prediction = classification_model.predict(img_processed)[0]
    class_idx = np.argmax(prediction)
    tumor_name = TUMOR_CLASSES[class_idx]
    confidence = prediction[class_idx] * 100
    return original_img, tumor_name, confidence

# **Segment tumor, draw contours & measure area/perimeter**
def segment_and_draw(image_path, original_img, tumor_name, confidence):
    _, img_processed = preprocess_image(image_path)
    pred_mask = segmentation_model.predict(img_processed)[0]

    # Convert mask to binary and resize to original image size
    mask = (pred_mask > 0.5).astype(np.uint8) * 255
    mask_resized = cv2.resize(mask, (original_img.shape[1], original_img.shape[0]))

    # Find contours of the tumor mask
    contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Convert grayscale to BGR
    result_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)

    # Initialize area & perimeter values
    total_area = 0
    total_perimeter = 0

    # Calculate area & perimeter for each contour
    for contour in contours:
        total_area += cv2.contourArea(contour)  # Area in square pixels (sq px)
        total_perimeter += cv2.arcLength(contour, True)  # Perimeter in pixels (px)

    # Draw classification result (tumor type & confidence)
    label_text = f"{tumor_name} ({confidence:.2f}%)"
    cv2.putText(result_img, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display area & perimeter at the bottom
    height = result_img.shape[0]
    metrics_text = f"Area: {total_area:.2f} sq px | Perimeter: {total_perimeter:.2f} px"
    cv2.putText(result_img, metrics_text, (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Draw red tumor contours
    cv2.drawContours(result_img, contours, -1, (0, 0, 255), 2)

    return result_img

# **Main function to process image**
def process_image(image_path):
    if not os.path.exists(image_path):
        print("❌ Error: Image path does not exist.")
        return

    original_img, tumor_name, confidence = predict_tumor(image_path)
    result_img = segment_and_draw(image_path, original_img, tumor_name, confidence)

    # Save the result
    os.makedirs("predicted_images", exist_ok=True)
    output_filename = os.path.join("predicted_images", f"{os.path.basename(image_path).split('.')[0]}_{tumor_name}.png")
    cv2.imwrite(output_filename, result_img)

# **Take image path as input**
image_path = input("Enter the image path: ").strip()
process_image(image_path)

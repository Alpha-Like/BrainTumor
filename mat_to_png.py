import os
import numpy as np
import h5py # type: ignore
from PIL import Image

input_folder = "1512427/brainTumorDataPublic_2299-3064"
output_folder = "dataset"

os.makedirs(os.path.join(output_folder, 'images'), exist_ok=True)
os.makedirs(os.path.join(output_folder, 'masks'), exist_ok=True)

for file in os.listdir(input_folder):
    if file.endswith(".mat"):
        file_path = os.path.join(input_folder, file)

        with h5py.File(file_path, 'r') as mat_data:
            cjdata = mat_data["cjdata"]

            # Extract Image
            image = np.array(cjdata["image"]).T.astype(np.uint8)

            # Extract Tumor Mask
            tumor_mask = (np.array(cjdata["tumorMask"]).T * 255).astype(np.uint8)

            # Save Image and Mask
            img_name = os.path.splitext(file)[0]
            Image.fromarray(image).save(os.path.join(output_folder, 'images', f"{img_name}.png"))
            Image.fromarray(tumor_mask).save(os.path.join(output_folder, 'masks', f"{img_name}.png"))

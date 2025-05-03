import cv2
import os
import numpy as np
import time
from tqdm import tqdm  # For progress bar
import matplotlib.pyplot as plt

# Define the paths
input_dataset_path = "Project/face/ck+"  
output_dataset_path = "Check/CK+_clahe"  

if not os.path.exists(output_dataset_path):
    os.makedirs(output_dataset_path)

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))

emotions = ["anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]

total_images = sum([len(os.listdir(os.path.join(input_dataset_path, e))) for e in emotions])

print("Starting CLAHE preprocessing...\n")
pbar = tqdm(total=total_images, desc="Processing Images", ncols=100, bar_format="{l_bar}{bar} {n_fmt}/{total_fmt} [Time: {elapsed}]")

for emotion in emotions:
    input_folder = os.path.join(input_dataset_path, emotion)
    output_folder = os.path.join(output_dataset_path, emotion)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = sorted(os.listdir(input_folder))

    for idx, img_name in enumerate(image_files, start=1):
        img_path = os.path.join(input_folder, img_name)
        output_img_path = os.path.join(output_folder, f"{idx}.png")

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue 

        clahe_image = clahe.apply(image)

        cv2.imwrite(output_img_path, clahe_image)

        if np.random.randint(0, 400) == 0:
            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(image, cmap="gray")
            plt.title("Original Image")
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.imshow(clahe_image, cmap="gray")
            plt.title("CLAHE Processed")
            plt.axis("off")

            plt.show()
            time.sleep(0.5)  # Small delay for effect

        pbar.update(1)  # Update progress bar

pbar.close()
print("\nCLAHE preprocessing completed for all CK+ images! 🚀")

import cv2
import os
import albumentations as A

# Define input and output directories
input_folder = "D:/gradproject/image_recognition/image_classification/data/images/orginal/Nefertiti"  # Folder containing the original images
output_folder = "D:/gradproject/image_recognition/image_classification/data/images/train/Nefertiti"  # Folder to save the augmented images
os.makedirs(output_folder, exist_ok=True)

# Check if input folder exists
if not os.path.exists(input_folder):
    raise FileNotFoundError(f"Input folder not found: {input_folder}")

# Define augmentations
augmentations = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.GaussNoise(p=0.3),
    A.Affine(scale=(0.8, 1.2), p=0.5),
])

# Process each image in the input folder
for file_name in os.listdir(input_folder):
    if file_name.endswith((".jpg", ".jpeg", ".png")):  # Filter for image files
        image_path = os.path.join(input_folder, file_name)
        
        # Check if file exists
        if not os.path.exists(image_path):
            print(f"File not found: {image_path}")
            continue

        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image: {image_path}")
            continue

        for i in range(10):  # Generate 10 augmentations per image
            augmented = augmentations(image=image)
            augmented_image = augmented['image']

            # Save augmented image
            output_file_name = f"{os.path.splitext(file_name)[0]}_augmented_{i}.jpg"
            output_path = os.path.join(output_folder, output_file_name)
            cv2.imwrite(output_path, augmented_image)

print(f"Augmented images saved to {output_folder}")

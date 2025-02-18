import os
import shutil
import random

# Paths to train and train_reduced directories
train_dir = 'C:/Users/user/Desktop/PPROJECT/implementation/train'
train_reduced_dir = 'C:/Users/user/Desktop/PPROJECT/implementation/train_reduced'

# Step 1: Clear everything in the train_reduced folder
def clear_train_reduced(train_reduced_dir):
    if os.path.exists(train_reduced_dir):
        shutil.rmtree(train_reduced_dir)  # Deletes train_reduced folder
    os.makedirs(train_reduced_dir)  # Creates a new empty train_reduced folder

# Step 2: Copy 500 random images from each letter folder in train to train_reduced
def copy_images(train_dir, train_reduced_dir, num_samples=500):
    # Iterate over each letter folder in the train directory
    for letter_folder in os.listdir(train_dir):
        letter_path = os.path.join(train_dir, letter_folder)
        
        if os.path.isdir(letter_path):  # Ensure it's a directory
            images = os.listdir(letter_path)
            random.shuffle(images)  # Shuffle the images to get a random selection
            
            # Limit to num_samples (500 by default)
            selected_images = images[:num_samples]
            
            # Create corresponding folder in train_reduced
            reduced_folder_path = os.path.join(train_reduced_dir, letter_folder)
            os.makedirs(reduced_folder_path, exist_ok=True)
            
            # Copy the selected images to train_reduced
            for img in selected_images:
                src_img_path = os.path.join(letter_path, img)
                dest_img_path = os.path.join(reduced_folder_path, img)
                shutil.copyfile(src_img_path, dest_img_path)
            print(f"Copied {len(selected_images)} images to {reduced_folder_path}")

# Clear train_reduced and then copy 500 images per folder
clear_train_reduced(train_reduced_dir)
copy_images(train_dir, train_reduced_dir, num_samples=500)

print("Done!")

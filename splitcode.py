import os
import shutil
import random

# Function to split data into train and test folders for each group
def split_data(source, destination_train, destination_test, split_ratio):
    for root, dirs, files in os.walk(source):
        for dir in dirs:
            if dir.startswith('.'):
                continue
            source_dir = os.path.join(root, dir)
            dest_dir_train = os.path.join(destination_train, dir)
            dest_dir_test = os.path.join(destination_test, dir)
            os.makedirs(dest_dir_train, exist_ok=True)
            os.makedirs(dest_dir_test, exist_ok=True)
            files = os.listdir(source_dir)
            num_files = len(files)
            num_test = int(num_files * split_ratio)
            test_files = random.sample(files, num_test)
            train_files = [file for file in files if file not in test_files]
            for file in test_files:
                shutil.move(os.path.join(source_dir, file), os.path.join(dest_dir_test, file))
            for file in train_files:
                shutil.move(os.path.join(source_dir, file), os.path.join(dest_dir_train, file))

# Function to create subfolders G6, G7, G8, and G10 within train_data folder
def create_group_folders(destination_train):
    groups = ["G6", "G7", "G8", "G10"]
    for group in groups:
        os.makedirs(os.path.join(destination_train, group), exist_ok=True)

# Paths to the original data and destination folders
source_folder = "/Users/sickle/Coding/Eason Li | Zishan Guo Research/ogdata"
train_data_folder = "/Users/sickle/Coding/Eason Li | Zishan Guo Research/train_data"
test_data_folder = "/Users/sickle/Coding/Eason Li | Zishan Guo Research/test_data"

# Split ratio for test data
split_ratio = 0.25

# Split the data for each group
for group_folder in os.listdir(source_folder):
    group_source_folder = os.path.join(source_folder, group_folder)
    group_train_folder = os.path.join(train_data_folder, group_folder)
    group_test_folder = os.path.join(test_data_folder, group_folder)
    split_data(group_source_folder, group_train_folder, group_test_folder, split_ratio)

# Create G6, G7, G8, and G10 folders within train_data folder
create_group_folders(train_data_folder)

# Move generated images to train_data_folder
generated_images_folders = ["4000_generated_images", "2500_generated_images"]
for folder in generated_images_folders:
    source_folder = os.path.join("/Users/sickle/Coding/Eason Li | Zishan Guo Research/", folder)
    dest_folder = os.path.join(train_data_folder, folder)
    shutil.move(source_folder, dest_folder)

print("Data split and organized successfully.")
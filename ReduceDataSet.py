import os
import shutil
import random
import time

import numpy as np
from ultralytics import YOLO

# Paths
original_data_dir = "cifar-10_data"
reduced_data_dir = "reduced_cifar10"
train_dir = os.path.join(original_data_dir, "train")
test_dir = os.path.join(original_data_dir, "test")


def create_reduced_dataset(original_dir, reduced_dir, fraction, run_num, subset_type):
    """
    Function to create a balanced reduced dataset, so an equal amount of images per class

    Input:
    - orginal_dir: path original dataset
    - reduced_dir: path where reduced dataset will be saved
    - fraction: retain fraction of images from each class (>0 to 1)
    - run_num: (unique) number to create disctinct path for different runs
    - subset_type : indicates type of subset ("train" or "test")
    """
    dataset_path = os.path.join(reduced_dir, f"run_{run_num}", subset_type)
    os.makedirs(dataset_path, exist_ok=True)

    for class_label in os.listdir(original_dir):
        class_path = os.path.join(original_dir, class_label)
        if not os.path.isdir(class_path):
            continue

        images = os.listdir(class_path)
        random.seed(time.time() + run_num) #to ensure randomness
        random.shuffle(images)
        num_images = int(len(images) * fraction)
        selected_images = images[:num_images]

        reduced_class_path = os.path.join(dataset_path, class_label)
        os.makedirs(reduced_class_path, exist_ok=True)

        for image in selected_images:
            src_path = os.path.join(class_path, image)
            dest_path = os.path.join(reduced_class_path, image)
            shutil.copy(src_path, dest_path)

def train_yolo(dataset_dir, image_size, epochs,patience):
    """
    Function to train a YOLO model

    Input:
    - dataset_dir : path to dataset
    - image_size: size image
    - epoch: number of epochs to train
    - patience: The number of epochs to wait for validation loss to improve before stopping early. If the validation
        loss does not improve after `patience` epochs, training will stop early to avoid overfitting.
    """
    model = YOLO("yolo11n-cls.pt")
    results = model.train(data=dataset_dir, imgsz=image_size, epochs=epochs, patience=patience, task="classification")
    accuracy = results.top1
    return accuracy


if __name__ == "__main__":
    fractions = [0.95] + [i / 10 for i in range(1, 10)] # Fractions of the dataset to test
    image_size = 32
    epochs = 50
    patience = 5

    for fraction in fractions:
        #Create path string
        fraction_str = str(fraction).replace(".", "_")
        fraction_dir = os.path.join(reduced_data_dir, f"fraction_{fraction_str}")
        accuracy_run = []
        for run_num in range(1, 6):  # Generate 5 random datasets per fraction

            print(f"Creating run {run_num} for fraction {fraction}.")

            # Create reduced training and testing datasets
            create_reduced_dataset(train_dir, fraction_dir, fraction, run_num, "train")
            create_reduced_dataset(test_dir, fraction_dir, fraction, run_num, "test")

            # Train the YOLO model
            run_path = os.path.join(fraction_dir, f"run_{run_num}")
            print(f"Training YOLO model for run {run_num} with fraction {fraction}.")
            accuracy = train_yolo(run_path, image_size, epochs, patience)
            accuracy_run.append(accuracy)
            print(f"Completed training for run {run_num} with fraction {fraction}, and accuracy {accuracy}.")

        accuracy_var = np.var(accuracy_run)
        accuracy_mean = np.mean(accuracy_run)

        #Print some interim results
        if fraction == 0.95:
            base_accuracy_var = accuracy_var
            base_accuracy_mean = accuracy_mean
            print(f"With fraction {fraction} (base); \nmean accuracy: {base_accuracy_mean}, \nvariance: {base_accuracy_var}")
        else:
            #base_accuracy_var = 0.0
            #base_accuracy_mean = 0.7867000102996826
            print(f"With fraction {fraction}; \nmean accuracy: {accuracy_mean}, \nvariance: {accuracy_var}")
            if abs((base_accuracy_mean-accuracy_mean)/base_accuracy_mean) < 0.1:
                break

    print(f"Investigation complete, best lowest found fraction: {fraction}.")

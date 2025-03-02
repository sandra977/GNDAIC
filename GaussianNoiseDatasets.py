import os
import shutil
import random
import time
import numpy as np

# Paths
original_data_dir = "cifar-10_data"
reduced_data_dir = "reduced_cifar10_research_V4"
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
        random.seed(time.time() + run_num)
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
        Function to train a YOLO model. Saves also confusion matrix as CSV

        Input:
        - dataset_dir : path to dataset
        - image_size: size image
        - epoch: number of epochs to train
        - patience: The number of epochs to wait for validation loss to improve before stopping early. If the validation
            loss does not improve after `patience` epochs, training will stop early to avoid overfitting.

        Output:
        - accuracy: The top-1 accuracy achieved after training.
        """
    model = YOLO("yolo11n-cls.pt")
    results = model.train(data=dataset_dir, imgsz=image_size, epochs=epochs, patience=patience, task="classification")
    accuracy = results.top1
    train_dir = results.save_dir

    # Get confusion matrix from the results
    cm = results.confusion_matrix
    cm = cm.matrix

    # Convert confusion matrix to pandas DataFrame
    cm_df = pd.DataFrame(cm)

    # Save confusion matrix as CSV in the training directory
    cm_df.to_csv(os.path.join(train_dir, 'confusion_matrix.csv'), index=False)  # Save as CSV in train directory

    return accuracy

def add_gaussian_noise(image, variance, mean=0, seed=None):
    """
    Adds Gaussian noise to an image.

    Input:
    - image: image to which gaussian noise should be added
    - variance: variance gaussian noise
    - mean: mean gaussian noise, default for these experiments is 0
    """
    sigma = variance ** 0.5  # Convert variance to standard deviation
    if seed is not None:
        np.random.seed(seed)
    # Generate noise as float32 (to prevent overflow/wrap-around issues)
    noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)

    # Add noise and ensure pixel values remain valid
    noisy_image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy_image

def create_extended_dataset(org_reduced_dir, ID_nr_dir, variance, nr_noise_img):
    """
    Function to create an extended dataset by adding noisy versions of images
    to an existing dataset.

    Input:
    - org_reduced_dir: path to original reduced dataet
    - ID_nr_dir: path to new directery
    - variance: variance of gaussian noise
    - nr_noise_img: number noisy versions of images to add

    """
    # Create train folder
    org_reduced_dir_train = os.path.join(org_reduced_dir, "train")
    new_class_path_train = os.path.join(ID_nr_dir, "train")
    os.makedirs(new_class_path_train, exist_ok=True)

    for class_label in os.listdir(org_reduced_dir_train):
        org_class_path = os.path.join(org_reduced_dir_train, class_label)
        print(org_class_path)
        if not os.path.isdir(org_class_path):
            continue

        new_class_path = os.path.join(new_class_path_train, class_label)
        os.makedirs(new_class_path, exist_ok=True)

        # Get image files
        image_files = [f for f in os.listdir(org_class_path)]

        for img_file in image_files:
            img_path = os.path.join(org_class_path, img_file)

            if not os.path.exists(img_path):
                continue

            # Read the image
            image = cv2.imread(img_path)
            if image is None:
                continue  # Skip unreadable files

            # Save the original image
            cv2.imwrite(os.path.join(new_class_path, img_file), image)

            # Generate and save noisy versions
            for i in range(nr_noise_img):
                noisy_img = add_gaussian_noise(image, variance, seed=int(time.time()) + i)
                noisy_img_name = f"{img_file.rsplit('.', 1)[0]}_noise{i}.jpg"
                cv2.imwrite(os.path.join(new_class_path, noisy_img_name), noisy_img)

    #Copy test folder
    org_reduced_dir_test = os.path.join(org_reduced_dir, "test")
    new_class_path_test = os.path.join(ID_nr_dir, "test")
    shutil.copytree(org_reduced_dir_test, new_class_path_test, dirs_exist_ok=True)



if __name__ == "__main__":

    image_size = 32
    epochs = 50
    patience = 5
    fraction = 0.1
    variance_vals = [10, 30, 50]
    num_generated_images = [5, 10, 15]

    ID_nr = 0
    ID_nr_dir = os.path.join(reduced_data_dir, "ID_nr_0")
    run_num = 1
    print(f"Creating base run for fraction {fraction}.")

    # Create reduced training and testing datasets
    create_reduced_dataset(train_dir, ID_nr_dir, fraction, run_num, "train")
    create_reduced_dataset(test_dir, ID_nr_dir, fraction, run_num, "test")

    # Train the YOLO model
    run_path = os.path.join(ID_nr_dir, f"run_{run_num}")
    print(f"Training YOLO model for base run with fraction {fraction}.")
    accuracy = train_yolo(run_path, image_size, epochs, patience)
    print(f"Completed training for base run with fraction {fraction}, and accuracy {accuracy}.")

    org_reduced_dir = os.path.join(reduced_data_dir, "ID_nr_0", "run_1")


    for var_val in variance_vals:
        for num_gen_img in num_generated_images:
            ID_nr +=1
            accuracy_run = []
            print(f"ID_nr:{ID_nr} and run_num{run_num}")
            for run_num in range(1,6):
                np.random.seed(int(time.time()) + run_num)
                ID_nr_dir = os.path.join(reduced_data_dir, f"ID_nr_{ID_nr}", f"run_{run_num}")
                #Create extended dataset
                create_extended_dataset(org_reduced_dir, ID_nr_dir, var_val, num_gen_img)
                #Train extended dataset
                print(f"Training YOLO model on reduced dataset (fraction={fraction}, and {num_gen_img} noisy images added with variance of {var_val}.")
                accuracy = train_yolo(ID_nr_dir, image_size, epochs, patience)
                accuracy_run.append(accuracy)
                print(f"Completed training (augmentation with {num_gen_img} noisy images and noise variance of {var_val}), gives accuracy {accuracy}.")
            #Used only for quick indication performance
            accuracy_var = np.var(accuracy_run)
            accuracy_mean = np.mean(accuracy_run)
            print(f"Augmentation with {num_gen_img} noisy images and noise variance of {var_val}; \nmean accuracy: {accuracy_mean}, \nvariance: {accuracy_var}")

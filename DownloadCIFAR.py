import os
import urllib.request
import tarfile
import pickle
import numpy as np
from PIL import Image

# URL for CIFA-10 and file paths where to save
cifar10_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
download_dir = "cifar-10_data"
compressed_file = os.path.join(download_dir, "cifar-10-python.tar.gz")
unpack_dir = os.path.join(download_dir, "cifar-10-batches-py")
train_folder = os.path.join(download_dir, "train")
test_folder = os.path.join(download_dir, "test")

# Ensure the download directory exists
os.makedirs(download_dir, exist_ok=True)

def download_cifar10():
    """Downloads the CIFAR-10 dataset, only if it was not downloaded yet."""
    if not os.path.exists(compressed_file):
        print("Downloading CIFAR-10 dataset.")
        urllib.request.urlretrieve(cifar10_url, compressed_file)
        print("Download complete.")
    else:
        print("CIFAR-10 dataset already downloaded.")

def extract_cifar10():
    """Extracts the CIFAR-10 dataset from the tar.gz file."""
    if not os.path.exists(unpack_dir):
        print("Extracting CIFAR-10 dataset.")
        with tarfile.open(compressed_file, "r:gz") as tar:
            tar.extractall(path=download_dir)
        print("Extraction complete.")
    else:
        print("CIFAR-10 dataset already extracted.")


def load_batch(file_path):
    """Loads a batch file from the CIFAR-10 dataset."""
    with open(file_path, "rb") as file:
        batch = pickle.load(file, encoding="bytes")
    data = batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    print(f"Loaded {len(data)} images from {file_path}")
    labels = batch[b"labels"]
    return data, labels

def save_images(data, labels, folder, batch_nr):
    """Saves the images to the specified folder, by labels."""
    os.makedirs(folder, exist_ok=True)
    print(f"Saving {len(data)} images to {folder}")
    for i, (image, label) in enumerate(zip(data, labels)):
        label_folder = os.path.join(folder, str(label))
        os.makedirs(label_folder, exist_ok=True)
        img = Image.fromarray(image)
        img.save(os.path.join(label_folder, f"img_{batch_nr}_{i}.png"))


def organize_cifar10():
    """Organizes the CIFAR-10 dataset into training and test folders."""
    print("Organizing CIFAR-10 dataset.")
    # Process training batches
    for batch_nr in range(1, 6):
        batch_file = os.path.join(unpack_dir, f"data_batch_{batch_nr}")
        data, labels = load_batch(batch_file)
        print(f"Processing {len(data)} images for batch {batch_file}")
        save_images(data, labels, train_folder, batch_nr)

    # Process test batch
    test_batch_file = os.path.join(unpack_dir, "test_batch")
    data, labels = load_batch(test_batch_file)
    save_images(data, labels, test_folder, 0)

    print("Train and test data have been organized into respective folders.")

if __name__ == "__main__":
    download_cifar10()
    extract_cifar10()
    organize_cifar10()
    print("CIFAR-10 dataset setup complete.")

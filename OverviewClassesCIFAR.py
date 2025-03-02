import pickle
import numpy as np
import matplotlib.pyplot as plt

# Function to unpickle a file
def unpickle(file):
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict

# Load the CIFAR-10 data
data_dir = 'Old/data/cifar-10/cifar-10-batches-py'

# Load meta information (class names)
meta = unpickle(f"{data_dir}/batches.meta")
class_names = [name.decode('utf-8') for name in meta[b'label_names']]

# Combine all training batches
def load_all_batches(data_dir):
    images, labels = [], []
    for i in range(1, 6):
        batch = unpickle(f"{data_dir}/data_batch_{i}")
        batch_images = batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        images.append(batch_images)
        labels += batch[b'labels']
    return np.concatenate(images), np.array(labels)

train_images, train_labels = load_all_batches(data_dir)

# Dictionary to store two images per class
class_images = {class_id: [] for class_id in range(10)}

# Collect two images per class
for image, label in zip(train_images, train_labels):
    if len(class_images[label]) < 2:  # Only add if we haven't yet collected two images
        class_images[label].append(image)
    if all(len(imgs) == 2 for imgs in class_images.values()):  # Stop when all classes are filled
        break

# Create the plot
fig, axes = plt.subplots(2, 10, figsize=(20, 5))
#fig.suptitle('Two Images from Each Class', fontsize=16)

# Plot images
for class_id, imgs in class_images.items():
    for i, img in enumerate(imgs):  # i is 0 for the first row, 1 for the second row
        ax = axes[i, class_id]
        ax.imshow(img)
        ax.axis('off')
        if i == 0:  # Add class name only to the top row
            ax.set_title(class_names[class_id], fontsize=28)

plt.tight_layout(rect=[0, 0, 1, 0.93])  # Adjust layout to fit the title
plt.show()

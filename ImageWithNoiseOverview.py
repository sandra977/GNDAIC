import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = "img_1_405.png"


def add_gaussian_noise(image, variance):
    mean = 0
    sigma = variance ** 0.5
    gauss = np.random.normal(mean, sigma, image.shape)

    noisy_image = np.clip(image + gauss, 0, 255).astype(np.uint8)
    return noisy_image


def create_images():
    gradient_image = np.tile(np.linspace(0, 255, 32, dtype=np.uint8), (32, 1))
    return gradient_image


# Load original image
image = cv2.imread(image_path, cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB for proper display

# Generate noisy images for original
noise_10 = add_gaussian_noise(image, 10)
noise_30 = add_gaussian_noise(image, 30)
noise_50 = add_gaussian_noise(image, 50)

# Generate grayscale images
gradient_image = create_images()
gradient_noise_10 = add_gaussian_noise(gradient_image, 10)
gradient_noise_30 = add_gaussian_noise(gradient_image, 30)
gradient_noise_50 = add_gaussian_noise(gradient_image, 50)

# Plot images
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# First row: original and noisy versions
axes[0, 0].imshow(image)
axes[0, 0].set_title("Original Image")
axes[0, 1].imshow(noise_10)
axes[0, 1].set_title("Original + Noise (Variance 10)")
axes[0, 2].imshow(noise_30)
axes[0, 2].set_title("Original +  Noise (Variance 30)")
axes[0, 3].imshow(noise_50)
axes[0, 3].set_title("Original +  Noise (Variance 50)")

# Second row: grayscale and noisy versions
axes[1, 0].imshow(gradient_image)
axes[1, 0].set_title("Gradient Image")
axes[1, 1].imshow(gradient_noise_10)
axes[1, 1].set_title("Gradient + Noise (Variance 10)")
axes[1, 2].imshow(gradient_noise_30)
axes[1, 2].set_title("Gradient + Noise (Variance 30)")
axes[1, 3].imshow(gradient_noise_50)
axes[1, 3].set_title("Gradient + Noise")

# Remove axes
for ax in axes.ravel():
    ax.axis("off")

plt.show()

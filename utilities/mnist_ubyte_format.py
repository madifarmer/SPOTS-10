import cv2
import numpy as np
import os
import glob

from PIL import Image
import gzip
import struct


def load_images_from_folders(base_folder, class_descriptions):
    images = []
    labels = []
    for label, class_name in enumerate(class_descriptions):
        folder_path = os.path.join(base_folder, class_name)
        print("Processing class: ", class_name)
        if not os.path.exists(folder_path):
            continue
        for filename in os.listdir(folder_path):
            if filename.endswith(".png"):
                img = Image.open(os.path.join(folder_path, filename)).convert('L')  # Convert image to grayscale
                img = img.resize((28, 28))  # Resize image to 28x28 pixels
                img_np = np.array(img, dtype=np.uint8)
                images.append(img_np)
                labels.append(label)
    return np.array(images), np.array(labels)


def save_idx_images(filename, images):
    with gzip.open(filename, 'wb') as f:
        # Write header
        f.write(struct.pack('>IIII', 2051, images.shape[0], images.shape[1], images.shape[2]))
        # Write data
        f.write(images.tobytes())


def save_idx_labels(filename, labels):
    with gzip.open(filename, 'wb') as f:
        # Write header
        f.write(struct.pack('>II', 2049, labels.shape[0]))
        # Write data
        f.write(labels.tobytes())


class_descriptions = [
    "cheetah", "deer", "giraffe", "hyena", "jaguar",
    "leopard", "tapir", "tiger", "WhaleShark", "zebra"
]

train_images, train_labels = load_images_from_folders('../dataset/train', class_descriptions)
test_images, test_labels = load_images_from_folders('../dataset/test', class_descriptions)

#print(train_images.shape)

save_idx_images('train-images-idx3-ubyte.gz', train_images)
save_idx_labels('train-labels-idx1-ubyte.gz', train_labels)
save_idx_images('t10k-images-idx3-ubyte.gz', test_images)
save_idx_labels('t10k-labels-idx1-ubyte.gz', test_labels)
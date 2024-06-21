import os
import numpy as np
from PIL import Image
import gzip
import struct


class MNISTUbyteImageProcessor:
    def __init__(self, base_folder, class_descriptions, image_size=(32, 32)):
        self.base_folder = base_folder
        self.class_descriptions = class_descriptions
        self.image_size = image_size

    def load_images_from_folders(self):
        images = []
        labels = []
        for label, class_name in enumerate(self.class_descriptions):
            folder_path = os.path.join(self.base_folder, class_name)
            print("Processing class: ", class_name)
            if not os.path.exists(folder_path):
                continue
            for filename in os.listdir(folder_path):
                if filename.endswith(".png"):
                    img = Image.open(os.path.join(folder_path, filename)).convert('L')  # Convert image to grayscale
                    img = img.resize(self.image_size)  # Resize image to 32x32 pixels
                    img_np = np.array(img, dtype=np.uint8)
                    images.append(img_np)
                    labels.append(label)
        return np.array(images), np.array(labels)

    @staticmethod
    def save_idx_images(filename, images):
        with gzip.open(filename, 'wb') as f:
            # Magic number: 2051 for images
            f.write(struct.pack('>I', 2051))
            # Number of images
            f.write(struct.pack('>I', images.shape[0]))
            # Rows and columns
            f.write(struct.pack('>I', images.shape[1]))
            f.write(struct.pack('>I', images.shape[2]))
            # Image data, pixel values
            for img in images:
                f.write(img.tobytes())

    '''def save_idx_images(filename, images):
        with gzip.open(filename, 'wb') as f:
            # Write header
            f.write(struct.pack('>IIII', 2051, images.shape[0], images.shape[1], images.shape[2]))
            # Write data
            f.write(images.tobytes())'''

    @staticmethod
    def save_idx_labels(filename, labels):
        with gzip.open(filename, 'wb') as f:
            # Magic number: 2049 for labels
            f.write(struct.pack('>I', 2049))
            # Number of labels
            f.write(struct.pack('>I', labels.shape[0]))
            # Label data
            f.write(bytes(labels))


    '''def save_idx_labels(filename, labels):
        with gzip.open(filename, 'wb') as f:
            # Write header
            f.write(struct.pack('>II', 2049, labels.shape[0]))
            # Write data
            f.write(labels.tobytes())'''


# Example implementation
if __name__ == "__main__":
    class_descriptions = [
        "cheetah", "deer", "giraffe", "hyena", "jaguar",
        "leopard", "tapir", "tiger", "WhaleShark", "zebra"
    ]

    # Instantiate the processor for training and testing data
    train_processor = MNISTUbyteImageProcessor('../dataset/train', class_descriptions, image_size=(32,32))
    test_processor = MNISTUbyteImageProcessor('../dataset/test', class_descriptions, image_size=(32,32))

    # Load images and labels
    train_images, train_labels = train_processor.load_images_from_folders()
    test_images, test_labels = test_processor.load_images_from_folders()

    print(test_images.shape, test_labels.shape)
    print(train_images.shape, train_labels.shape)

    # Save images and labels in IDX format
    train_processor.save_idx_images('../dataset/train-images-idx3-ubyte.gz', train_images)
    train_processor.save_idx_labels('../dataset/train-labels-idx1-ubyte.gz', train_labels)
    test_processor.save_idx_images('../dataset/t10k-images-idx3-ubyte.gz', test_images)
    test_processor.save_idx_labels('../dataset/t10k-labels-idx1-ubyte.gz', test_labels)

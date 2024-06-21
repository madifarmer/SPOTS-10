import gzip
import numpy as np
import struct


class SPOT10Loader:
    def __init__(self, dataset_dir="."):
        self.dataset_dir = dataset_dir
        self.file_train_images = "train-images-idx3-ubyte.gz"
        self.file_train_labels = "train-labels-idx1-ubyte.gz"
        self.file_test_images = "t10k-images-idx3-ubyte.gz"
        self.file_test_labels = "t10k-labels-idx1-ubyte.gz"

    def load_labels_idx(self, filename):
        try:
            with gzip.open(filename, 'rb') as f:
                # Read magic number, number of items
                magic_number, num_items = struct.unpack('>II', f.read(8))
                # Check if it's the correct magic number for labels (2049)
                if magic_number != 2049:
                    raise ValueError(f'Invalid magic number {magic_number} for labels file {filename}')

                # Read labels data
                labels = np.frombuffer(f.read(num_items), dtype=np.uint8)

            return labels

        except FileNotFoundError:
            raise FileNotFoundError(f"File '{filename}' not found.")
        except Exception as e:
            raise RuntimeError(f"Error loading file '{filename}': {str(e)}")


    def load_images_idx(self, filename):
        try:
            with gzip.open(filename, 'rb') as f:
                # Read magic number, number of images, rows, and columns
                magic_number, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
                # Check if it's the correct magic number for images (2051)
                if magic_number != 2051:
                    raise ValueError(f'Invalid magic number {magic_number} for images file {filename}')

                # Read image data
                image_data = np.frombuffer(f.read(num_images * rows * cols), dtype=np.uint8)
                images = image_data.reshape(num_images, rows, cols)

            return images

        except FileNotFoundError:
            raise FileNotFoundError(f"File '{filename}' not found.")
        except Exception as e:
            raise RuntimeError(f"Error loading file '{filename}': {str(e)}")

    def get_data(self, dataset="train"):
        dataset = dataset.lower()
        if dataset == "train":
            images = self.load_images_idx(self.dataset_dir + "/" + self.file_train_images)
            labels = self.load_labels_idx(self.dataset_dir + "/" + self.file_train_labels)
            return images, labels
        elif dataset == "test":
            images = self.load_images_idx(self.dataset_dir + "/" + self.file_test_images)
            labels = self.load_labels_idx(self.dataset_dir + "/" + self.file_test_labels)
            return images, labels
        else:
            raise ValueError(f"Dataset type '{dataset}' not supported. Use 'train' or 'test'.")


# Example implementation
if __name__ == "__main__":
    data_loader = SPOT10Loader(dataset_dir="../dataset")
    data, labels = data_loader.get_data(dataset="Train")
    print(data.shape, labels.shape)
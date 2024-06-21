import gzip
import numpy as np
import struct


class SPOT10Loader:

    def __init__(self):
        pass

    @staticmethod
    def load_labels_idx(filename):
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

    @staticmethod
    def load_images_idx(filename):
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

    @staticmethod
    def get_data(dataset_dir="../dataset", kind="train"):
        kind = kind.lower()
        if kind == "train":
            images = SPOT10Loader.load_images_idx(dataset_dir + "/train-images-idx3-ubyte.gz")
            labels = SPOT10Loader.load_labels_idx(dataset_dir + "/train-labels-idx1-ubyte.gz")
            return images, labels
        elif kind == "test":
            images = SPOT10Loader.load_images_idx(dataset_dir + "/t10k-images-idx3-ubyte.gz")
            labels = SPOT10Loader.load_labels_idx(dataset_dir + "/t10k-labels-idx1-ubyte.gz")
            return images, labels
        else:
            raise ValueError(f"Dataset type '{kind}' not supported. Use 'train' or 'test'.")


# Example implementation
if __name__ == "__main__":
    data_loader = SPOT10Loader()
    data, labels = data_loader.get_data(kind="train")
    print(data.shape, labels.shape)

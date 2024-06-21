# SPOTS-10
The SPOTS-10 dataset was created to evaluate machine learning algorithms. This dataset consists of a comprehensive collection of grayscale images showcasing the diverse patterns found in ten animal species. Specifically, SPOTS-10 features 32 × 32 grayscale images of 50,000 distinct markings of ten animals species. The dataset is divided into ten categories, with 5,000 images per category. The training set comprises 40,000 images, while the test
set contains 10,000 images. 

Below are samples taken from the SPOTS-10 dataset were each row represents a category in the dataset and columns, variations of the samples in that category.

# Getting the Data
You can get the SPOTS-10 dataset by cloning this GitHub repository; the dataset appears under /dataset. This repo also contains some scripts for benchmark and the utilities folder that contains the files we used for making the MNIST-like dataset for SPOTS-10. You can also find some scripts that will help you load the train and test dataset and labels into a numpy array for training your benchmark model.

    git clone git@github.com:Amotica/SPOTS-10.git 

# Categories (Labels)
Each training and test samples is assigned to one of the following categories/labels:

Label	Description
0	T-shirt/top
1	Trouser
2	Pullover
3	Dress
4	Coat
5	Sandal
6	Shirt
7	Sneaker
8	Bag
9	Ankle boot

# Usage

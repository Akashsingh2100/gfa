import os
import random
import yaml
import tensorflow as tf
from sklearn import preprocessing
from sklearn.utils import shuffle
from tqdm import tqdm


class Dataset:
    def __init__(self, name, batch_size, attack_dir, real_dir):  # Fixed __init__
        self.name = name
        self.batch_size = batch_size
        self.attack_dir = attack_dir
        self.real_dir = real_dir
        self.dataset = MsuMsfdDataset(batch_size, attack_dir, real_dir)

        file_path, label_truth = self.dataset.load_idx()

        # Encode the labels
        self.encoding_truth = preprocessing.LabelEncoder()
        self.encoding_truth.fit(label_truth)
        self.list_label_truth = self.encoding_truth.transform(label_truth)

        self.list_file_path_truth = file_path.copy()

        self.shuffle_dataset()
        self.len_dataset = len(self.list_file_path_truth)
    

        with open("config.yaml", "r") as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
        self.standard_img_size = cfg['net']['input_img_size']

        # Build dataset using tf.data for better performance
        self.train_dataset = self.build_tf_dataset()

    def shuffle_dataset(self):
        self.list_file_path_truth, self.list_label_truth = shuffle(self.list_file_path_truth, self.list_label_truth,
                                                                   random_state=10)

    def load_and_preprocess_image(self, file_path, label):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, (self.standard_img_size, self.standard_img_size))
        img = img / 255.0  # Normalize to [0, 1]
        return img, label

    def augment_image(self, img, label):
        # Apply random flipping, brightness changes, etc.
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.1)
        return img, label

    def build_tf_dataset(self):
        # Create dataset from file paths and labels
        dataset = tf.data.Dataset.from_tensor_slices((self.list_file_path_truth, self.list_label_truth))
        dataset = dataset.shuffle(buffer_size=self.len_dataset)  # Shuffle dataset
        dataset = dataset.map(self.load_and_preprocess_image, num_parallel_calls=2)  # Load images
        dataset = dataset.map(self.augment_image, num_parallel_calls=2)  # Data augmentation
        dataset = dataset.batch(self.batch_size)  # Batch the data
        dataset = dataset.prefetch(buffer_size=2)  # Prefetch for optimal performance
        return dataset

    def generate_minibatch(self):
        start_idx = 0
        total_batches = (self.len_dataset + self.batch_size - 1) // self.batch_size
        print("Using tf.data pipeline for minibatch generation")

        with tqdm(total=total_batches, desc="Generating Minibatches", unit="batch") as pbar:
            for batch in self.train_dataset:
                batch_img_4_truth, batch_label_truth = batch

                # Select random images for LPC loss
                batch_random_1 = []
                batch_random_2 = []
                list_random_images_path1 = random.sample(self.list_file_path_truth, k=len(batch_img_4_truth))
                list_random_images_path2 = random.sample(self.list_file_path_truth, k=len(batch_img_4_truth))

                for file_path_1 in list_random_images_path1:
                    img1 = self.load_and_preprocess_image(file_path_1, None)[0]
                    batch_random_1.append(img1)

                for file_path_2 in list_random_images_path2:
                    img2 = self.load_and_preprocess_image(file_path_2, None)[0]
                    batch_random_2.append(img2)

                batch_img_random_1 = tf.stack(batch_random_1)
                batch_img_random_2 = tf.stack(batch_random_2)
                batch_label_random = tf.zeros(len(batch_img_random_1), dtype=tf.float32)

                start_idx += self.batch_size
                pbar.update(1)
                yield batch_img_4_truth, batch_label_truth, batch_img_random_1, batch_img_random_2, batch_label_random


class MsuMsfdDataset:
    def __init__(self, batch_size, attack_dir, real_dir):  # Fixed __init__
        self.batch_size = batch_size
        self.attack_dir = attack_dir
        self.real_dir = real_dir

    def load_idx(self):
        list_file_path = []
        list_label_truth = []
        attack_dir = self.attack_dir
        real_dir = self.real_dir

        print('Loading index for attack samples...')
        for pic in os.listdir(attack_dir):
            list_file_path.append(os.path.join(attack_dir, pic))
            list_label_truth.append('attack')

        print('Loading index for real samples...')
        for pic in os.listdir(real_dir):
            list_file_path.append(os.path.join(real_dir, pic))
            list_label_truth.append('real')

        return list_file_path, list_label_truth

if __name__ == '__main__':
    msumsfd_dataset = Dataset('replayattack', 32, r"E:\akash singh\GFA-CNN-new\GFA-CNN-master\GFA-CNN-master\dataset_small\msumsfd_gfacnn\attack",
                              real_dir=r"E:\akash singh\GFA-CNN-new\GFA-CNN-master\GFA-CNN-master\dataset_small\msumsfd_gfacnn\real")

    for img_4_truth, label_truth, img_random_1, img_random_2, label_random in msumsfd_dataset.generate_minibatch():
        print(label_truth)



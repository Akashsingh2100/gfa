import os
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.utils import shuffle
import cv2
import numpy as np
import yaml
import random
import matplotlib.pyplot as plt

class Dataset:
    def __init__(self, name, batch_size, attack_dir, real_dir):
        self.name = name
        self.batch_size = batch_size
        self.attack_dir = attack_dir
        self.real_dir = real_dir
        self.dataset = MsuMsfdDataset(batch_size, attack_dir, real_dir)

        file_path, label_truth = self.dataset.load_idx()
        
        # encode the label
        self.encoding_truth = preprocessing.LabelEncoder()
        self.encoding_truth.fit(label_truth)
        self.list_label_truth = self.encoding_truth.transform(label_truth)

        self.list_file_path_truth = file_path.copy()

        self.shuffle_dataset()
        self.len_dataset = len(self.list_file_path_truth)

        with open("E:/akash singh/GFA-CNN-new/GFA-CNN-master/GFA-CNN-master/config.yaml", "r") as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
        self.standard_img_size = cfg['net']['input_img_size']

    def shuffle_dataset(self):
        self.list_file_path_truth, self.list_label_truth = shuffle(self.list_file_path_truth, self.list_label_truth,
                                                                   random_state=10)

    def generate_minibatch(self):
        start_idx = 0
        total_batches = (self.len_dataset + self.batch_size - 1) // self.batch_size
        print("using minibatch")

        with tqdm(total=total_batches, desc="Generating Minibatches", unit="batch") as pbar:
            while True:
                # crop sub lists from the long lists
                if start_idx + self.batch_size <= self.len_dataset:
                    batch_file_path_truth = self.list_file_path_truth[start_idx: start_idx + self.batch_size]
                    batch_label_truth = self.list_label_truth[start_idx: start_idx + self.batch_size]
                    print(f"start_idx:{start_idx}, len_dataset: {self.len_dataset}, batch_size: {self.batch_size}")
                elif start_idx < self.len_dataset < (start_idx + self.batch_size):
                    batch_file_path_truth = self.list_file_path_truth[start_idx: self.len_dataset]
                    batch_label_truth = self.list_label_truth[start_idx: self.len_dataset]
                elif start_idx >= self.len_dataset:
                    break

                # load image to numpy array
                batch_img_4_truth = None
                for file_path_truth in batch_file_path_truth:
                    img = cv2.imread(file_path_truth)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (self.standard_img_size, self.standard_img_size))
                    img = np.expand_dims(img, axis=0)
                    img = img / 255.0  # Scale image to [0, 1]
                    if batch_img_4_truth is None:
                        batch_img_4_truth = img
                    else:
                        batch_img_4_truth = np.concatenate((batch_img_4_truth, img), axis=0)

                # select random images for lpc loss
                batch_random_1 = []
                batch_random_2 = []
                list_random_images_path1 = random.sample(self.list_file_path_truth, k=len(batch_img_4_truth))
                list_random_images_path2 = random.sample(self.list_file_path_truth, k=len(batch_img_4_truth))
                for file_path_1 in list_random_images_path1:
                    img = cv2.imread(file_path_1)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (self.standard_img_size, self.standard_img_size))
                    img = img / 255.0  # Scale image to [0, 1]
                    batch_random_1.append(img)

                for file_path_2 in list_random_images_path2:
                    img2 = cv2.imread(file_path_2)
                    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                    img2 = cv2.resize(img2, (self.standard_img_size, self.standard_img_size))
                    img2 = img2 / 255.0  # Scale image to [0, 1]
                    batch_random_2.append(img2)

                batch_img_random_1 = np.array(batch_random_1)
                batch_img_random_2 = np.array(batch_random_2)
                batch_label_random = [0 for i in range(len(batch_img_random_1))]
                batch_label_random = np.asarray(batch_label_random).astype(float)

                start_idx += self.batch_size
                pbar.update(1)
                yield batch_img_4_truth, batch_label_truth, batch_img_random_1, batch_img_random_2, batch_label_random

    # def generate_batch(self):
    #     img_4_truth = None
    #     label_truth = None
    #     img_random_1 = None
    #     img_random_2 = None
    #     label_random = None

    #     for batch_img_4_truth, batch_label_truth, batch_img_random_1, batch_img_random_2, batch_label_random in self.generate_minibatch():
    #         if img_4_truth is not None:
    #             img_4_truth = np.concatenate((img_4_truth, batch_img_4_truth), axis=0)
    #         else:
    #             img_4_truth = batch_img_4_truth

    #         if label_truth is not None:
    #             label_truth = np.concatenate((label_truth, batch_label_truth), axis=0)
    #         else:
    #             label_truth = batch_label_truth

    #         if img_random_1 is not None:
    #             img_random_1 = np.concatenate((img_random_1, batch_img_random_1), axis=0)
    #         else:
    #             img_random_1 = batch_img_random_1

    #         if img_random_2 is not None:
    #             img_random_2 = np.concatenate((img_random_2, batch_img_random_2), axis=0)
    #         else:
    #             img_random_2 = batch_img_random_2

    #         if label_random is not None:
    #             label_random = np.concatenate((label_random, batch_label_random), axis=0)
    #         else:
    #             label_random = batch_label_random

    #     return img_4_truth / 255.0, label_truth.astype(int), img_random_1 / 255.0, img_random_2 / 255.0, label_random.astype(int)

class MsuMsfdDataset:
    def __init__(self, batch_size, attack_dir, real_dir):
        self.batch_size = batch_size
        self.attack_dir = attack_dir
        self.real_dir = real_dir

    def load_idx(self):
        list_file_path = []
        list_label_truth = []
        attack_dir = self.attack_dir
        real_dir = self.real_dir
        
        print('loading index for attack')
        for pic in os.listdir(attack_dir):
                list_file_path.append(os.path.join(attack_dir, pic))
                list_label_truth.append('attack')
        print('loading index for real')
        for pic in os.listdir(real_dir):
                list_file_path.append(os.path.join(real_dir, pic))
                list_label_truth.append('real')
        return list_file_path, list_label_truth

#if __name__ == '__main__':
    
    # msumsfd_dataset = Dataset('replayattack', 32, attack_dir=r"/home/ml/akash singh/GFA-CNN-new/GFA-CNN-master/GFA-CNN-master/dataset1/msumsfd_gfacnn_test/attack",
    #                             real_dir=r"/home/ml/akash singh/GFA-CNN-new/GFA-CNN-master/GFA-CNN-master/dataset1/msumsfd_gfacnn_test/real")
    
    # img_4_truth, label_truth, img_random_1, img_random_2, label_random = msumsfd_dataset.generate_batch()

    # # Debugging: Save the images and plot them
    # for i, img in enumerate(img_4_truth[:4]):  # Save only first 4 images for debugging
    #     path = os.path.join('debug_as', f'img_{i}.jpg')
    #     cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # h, w = 10, 10  # for raster image
    # nrows, ncols = 2, 2  # array of sub-plots
    # figsize = [6, 8]
    # fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    # # plot simple raster image on each sub-plot
    # for i, axi in enumerate(ax.flat):
    #     img = img_4_truth[i]
    #     axi.imshow(img)
    #     axi.set_title(label_truth[i])

    # plt.tight_layout()
    # plt.show()
from time import time
import random

import keras
import cv2
import numpy as np

from segmentation_project.dataset import Dataset
from segmentation_project.augmentation import Augmentation


class DataGenerator(keras.utils.Sequence):

    def __init__(self, categories, path_to_annotation, path_to_images, image_size, batch_size, valid=True):

        self.dataset = Dataset(categories=categories, path_to_annotation=path_to_annotation,
                               path_to_images=path_to_images)
        self.augmentation = Augmentation(img_size=image_size)
        self.sequential = self.augmentation.get_sequential(valid)

        self.categories = categories
        self.image_size = image_size
        self.batch_size = batch_size

        self.valid = valid
        self.shuffle = not valid
        self.on_epoch_end()


    def create_category_mask(self, polygons, width, height):
        mask_base = np.zeros((height, width), np.uint8)

        if len(polygons) != 0:
            for polygon in polygons:
                cv2.fillPoly(mask_base, polygon, 1)

        return mask_base

    def create_mask(self, labels, categories, width, height):
        masks = []
        for i, category in enumerate(categories):
            mask = self.create_category_mask(labels[category], width, height)
            masks.append(mask)

        categories_mask = np.stack(masks, axis=-1).astype('float')
        background = 1 - categories_mask.sum(axis=-1, keepdims=True)
        final_mask = np.concatenate((categories_mask, background), axis=-1).astype(np.uint8)
        return final_mask

    def generate_batch(self, indexes):

        images_batch = np.zeros((self.batch_size, self.image_size[0], self.image_size[1], 3))
        labels_batch = np.zeros((self.batch_size, self.image_size[0], self.image_size[1], len(self.categories)+1))

        images_path, labels = self.dataset.get_batch(indexes)

        for i, data in enumerate(zip(images_path, labels)):
            image_path, label = data
            image = cv2.imread(image_path)

            try:
                shape = image.shape
            except AttributeError as e:
                if 'None' in str(e):
                    _, image_name = image_path.rsplit('/', 1)
                    ind = self.dataset.image_name_list.index(image_name)
                    print('Image %s broken' % image_path)
                    rind = random.randint(0, len(self.dataset.image_name_list))
                    new_img_name = self.dataset.image_name_list[rind]
                    self.dataset.image_name_list[ind] = self.dataset.image_name_list[0]
                    print('Image %s is replaced by %s' % (image_name, new_img_name))
                    continue

            mask = self.create_mask(label, self.categories, width=shape[1], height=shape[0])

            segmap = self.augmentation.get_segmentation_map(segmap=mask, nb_classes=len(self.categories)+1, shape=shape)
            image_aug, mask_aug = self.sequential(image=image, segmentation_maps=segmap)
            mask_aug = mask_aug.get_arr_int()

            '''from visualization import Visualisator
            vis = Visualisator()
            vis.draw_masks(image=image_aug, mask=mask_aug, label=[1, 2, 3])
            cv2.imshow('image', image_aug)
            cv2.waitKey(0)'''


            images_batch[i] = image_aug
            labels_batch[i] = mask_aug


        return images_batch, labels_batch

    def __getitem__(self, item):
        st = time()
        indexes = [i + item * self.batch_size for i in range(self.batch_size)]
        image_batch, label_batch = self.generate_batch(indexes)
        #print(time() - st)
        return image_batch, label_batch

    def __len__(self):
        return int(len(self.dataset) / self.batch_size)

    def on_epoch_end(self):
        if self.shuffle:
            self.dataset.shuffle()


if __name__ == "__main__":
    datagenerator = DataGenerator(categories=['cat', 'dog'], path_to_annotation='instances_train2017.json',
                                  path_to_images='../dataset/train', image_size=[256, 256], batch_size=8, valid=True)

    for a, b in datagenerator:
        pass

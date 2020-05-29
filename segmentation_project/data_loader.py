import os

import requests
from pycocotools.coco import COCO
from tqdm import tqdm
import cv2


class DataLoader:
    def __init__(self, path_to_annotation, save_dir=''):
        self.coco = COCO(path_to_annotation)
        self.save_dir = save_dir

    def get_categories_id(self, categories_name):
        return self.coco.getCatIds(catNms=categories_name)

    def get_images_id(self, categories_id):
        return self.coco.getImgIds(catIds=categories_id)

    def get_annotation_ids(self, categories_id):
        return self.coco.getAnnIds(catIds=categories_id)

    def get_annotation(self, annotation_ids):
        return self.coco.loadAnns(annotation_ids)

    def get_images_info(self, images_id):
        return self.coco.loadImgs(images_id)

    def get_image(self, url):
        return requests.get(url).content

    def get_image_name(self, image_info):
        return image_info['file_name']

    def get_url(self, image_info):
        return image_info['coco_url']

    def check_image(self, image_name):
        return cv2.imread(os.path.join(self.save_dir, image_name))

    def delete_image(self, image_name):
        os.remove(os.path.join(self.save_dir, image_name))
        print('Broken image %s deleted' % image_name)

    def save_image(self, image, image_name):
        os.makedirs(self.save_dir, exist_ok=True)

        with open(os.path.join(self.save_dir, image_name), 'wb') as f:
            f.write(image)

    def download_image(self, categories_name):
        categories_id = self.get_categories_id(categories_name)

        for category_id in categories_id:
            images_id = self.get_images_id(category_id)
            images_info = self.get_images_info(images_id)

            for image_info in tqdm(images_info, "Download image"):
                image_name = self.get_image_name(image_info)
                url = self.get_url(image_info)

                image = self.get_image(url)
                self.save_image(image, image_name)

                status = self.check_image(image_name)
                if status is None:
                    self.delete_image(image_name)



        print('Images download finished')

    def get_segmentation(self, categories_name):
        categories_id = self.get_categories_id(categories_name)

        for category_id in categories_id:
            annotation_id = self.get_annotation_ids(category_id)
            annotations = self.get_annotation(annotation_id)

            for annotation in annotations:
                segmentation = annotation['segmentation']
                image_id = annotation['image_id']



if __name__ == "__main__":
    loader = DataLoader(path_to_annotation='instances_val2017.json', save_dir='../dataset/val')
    loader.get_segmentation('cat')

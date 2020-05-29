import os
import random

from segmentation_project.data_loader import DataLoader
from segmentation_project.annotation import Annotation


class Dataset:

    def __init__(self, categories, path_to_annotation, path_to_images):
        self.categories = categories
        self.data_loader = DataLoader(path_to_annotation=path_to_annotation, save_dir=path_to_images)
        self.annotation = Annotation(categories=categories)
        self.path_to_images = path_to_images
        self.image_name_list = None
        self.check_data_existence()
        self._generate_annotation()

    def __len__(self):
        return len(self.image_name_list)

    def shuffle(self):
        random.shuffle(self.image_name_list)

    def download_data(self):
        self.data_loader.download_image(self.categories)
        self._generate_annotation()
        self._fill_image_name_list()

    def check_data_existence(self):
        if os.path.isdir(self.path_to_images):
            images_name = os.listdir(self.path_to_images)

            if len(images_name) > 0:
                self.image_name_list = images_name
            else:
                self.download_data()
        else:
            self.download_data()

    def get_batch(self, indexes):
        """
        :param indexes: list of index
        :return: images_path - list of path, labels - list of dict {category: [data]}
        """

        images_path = []
        labels = []

        for ind in indexes:
            image_name = self.image_name_list[ind]
            data = self.annotation.get_data_by_name(image_name)

            images_path.append(os.path.join(self.path_to_images, image_name))
            labels.append(data)

        return images_path, labels

    def _fill_image_name_list(self):
        self.image_name_list = self.annotation.get_images_name()

    def _generate_annotation(self):
        for category in self.categories:

            category_id = self.data_loader.get_categories_id(category)[0]
            annotation_id = self.data_loader.get_annotation_ids(category_id)
            annotations = self.data_loader.get_annotation(annotation_id)

            for annotation in annotations:
                image_id = annotation['image_id']
                if annotation['iscrowd'] != 0:
                    segmentation = [annotation['segmentation']['counts']]
                else:
                    segmentation = annotation['segmentation']

                image_info = self.data_loader.get_images_info(image_id)[0]
                image_name = self.data_loader.get_image_name(image_info)

                self.annotation.add_element(image_name, category, segmentation)








if __name__ == "__main__":
    dataset = Dataset(['cat', 'dog'], 'instances_val2017.json', './dataset/val')
    dataset.generate_annotation()
    dataset._get_image_name_list()
    dataset.shuffle()
    a, b = dataset.get_batch([1, 2, 3, 4, 5])
    print()
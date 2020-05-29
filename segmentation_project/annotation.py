import numpy as np

class Annotation:

    def __init__(self, categories):
        self._categories = categories
        self._annotation = {}  # {image_name: {category: [segmentation]}}

    def create_polygon(self, data_list):
        polygons = []
        for i, data in enumerate(data_list):
            try:
                polygons.append(np.array([(int(data[ind]), int(data[ind + 1])) for ind in range(0, len(data), 2)]))
            except IndexError:
                polygons.append(np.array([(int(data[ind]), int(data[ind + 1])) for ind in range(0, len(data[:-1]), 2)]))
            return polygons

    def add_element(self, image_name, category, data):
        polygons = self.create_polygon(data)
        if image_name in self._annotation:
            self._annotation[image_name][category].append(polygons)
        else:
            self._annotation[image_name] = {cat: [] for cat in self._categories}
            self._annotation[image_name][category].append(polygons)

    def get_images_name(self):
        return list(self._annotation.keys())

    def get_data_by_name(self, image_name):
        return self._annotation[image_name]

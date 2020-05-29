from segmentation_project.data_generator import DataGenerator
from segmentation_project.model_builder import SegmentationModel


class Pipeline:
    def __init__(self, model_name, backbone_name, categories, image_size,
                 batch_size, path_to_train_annotation, path_to_train_images,
                 path_to_val_annotation, path_to_val_images):

        self.train_gen = DataGenerator(categories, path_to_train_annotation, path_to_train_images,
                                       image_size, batch_size, valid=False)

        self.val_gen = DataGenerator(categories, path_to_val_annotation, path_to_val_images,
                                     image_size, batch_size, valid=True)

        self.model = SegmentationModel(model_name, backbone_name, input_shape=image_size, classes=len(categories)+1)

    def train(self, lr, epochs):
        self.model.train(lr, self.train_gen, self.val_gen, epochs)

import cv2
import numpy as np


from segmentation_models.losses import categorical_focal_dice_loss
from segmentation_models.metrics import iou_score, precision, recall

from keras.utils.generic_utils import get_custom_objects

get_custom_objects().update({"focal_loss_plus_dice_loss": categorical_focal_dice_loss})
get_custom_objects().update({"iou_score": iou_score})
get_custom_objects().update({"precision": precision})
get_custom_objects().update({"recall": recall})

import mlflow.keras


from services.tracking.remote_registry import RemoteRegistry

class SegmentationModel:
    def __init__(self, tracking_uri, model_name):

        self.registry = RemoteRegistry(tracking_uri=tracking_uri)
        self.model_name = model_name
        self.model = self.build_model(model_name)

    def get_latest_model(self, model_name):
        registered_models = self.registry.get_registered_model(model_name)
        last_model = self.registry.get_last_model(registered_models)
        local_path = self.registry.download_artifact(last_model.run_id, 'model', './')
        return local_path

    def build_model(self, model_name):
        local_path = self.get_latest_model(model_name)

        return mlflow.keras.load_model(local_path)

    def predict(self, image):
        image = self.preprocess(image)
        result = self.model.predict(image)
        return self.postprocess(result)

    def preprocess(self, image):
        image = cv2.resize(image, (256, 256))
        image = image / 255.
        image = np.expand_dims(image, 0)
        return image

    def postprocess(self, result):
        return result

if __name__ == "__main__":
    model = SegmentationModel(tracking_uri='http://10.22.12.24:8000', model_name='MyModel')
    image = cv2.imread('cat.jpg')
    result = model.predict(image)
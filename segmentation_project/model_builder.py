import os

from segmentation_models import Unet, PSPNet, Linknet, FPN
import segmentation_models as sm

from segmentation_models.metrics import iou_score, precision, recall
from segmentation_models.losses import categorical_focal_dice_loss

from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam


class SegmentationModel:
    MODELS = {'Unet': Unet, 'PSPNet': PSPNet, 'Linknet': Linknet, 'FPN': FPN}

    def __init__(self, model_name, backbone_name, input_shape, classes,
                 activation='softmax', encoder_weights='imagenet'):

        self.model_name = model_name
        self.model = self.get_model_by_name(model_name)
        self.backbone_name = backbone_name
        self.input_shape = input_shape
        self.classes = classes
        self.encoder_weights = encoder_weights
        self.activation = activation

    def get_model_by_name(self, model_name):
        return self.MODELS[model_name]

    def _build_model(self, lr):
        self.model = self.model(backbone_name=self.backbone_name, input_shape=self.input_shape, classes=self.classes,
                                encoder_weights=self.encoder_weights, activation=self.activation)

        optimizer = Adam(lr=lr)
        self.model.compile(optimizer, categorical_focal_dice_loss, [iou_score, precision, recall])

    def train(self, lr, train_gen, val_gen, epochs, workers=4, use_multiprocessing=False):
        self._build_model(lr)

        self.model.fit_generator(train_gen,
                                 validation_data=val_gen,
                                 epochs=epochs,
                                 workers=workers,
                                 use_multiprocessing=use_multiprocessing)

if __name__ == "__main__":
    print(sm.framework())

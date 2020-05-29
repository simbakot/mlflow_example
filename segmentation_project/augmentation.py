from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapOnImage


class Augmentation:

    def __init__(self, img_size):
        self.img_size = img_size

    def get_segmentation_map(self, segmap, nb_classes, shape):
        return SegmentationMapOnImage(segmap, nb_classes=nb_classes, shape=shape)


    def get_sequential(self, valid=False):
        if valid:
            sequential = iaa.Sequential([
                iaa.Resize({"height": self.img_size[0], "width": self.img_size[1]})])
            return sequential

        sequential = iaa.Sequential([
            iaa.Sometimes(0.5,
                          iaa.Crop(percent=([0.05, 0.1], [0.05, 0.1], [0.05, 0.1], [0.05, 0.1])),
                          iaa.Fliplr(0.5)),

            iaa.Sometimes(0.3,
                          iaa.Affine(rotate=(-10, 10))),

            iaa.SomeOf((0, 3), [
                iaa.Sometimes(0.5,
                              iaa.EdgeDetect(alpha=(0.0, 0.5)),
                              iaa.ContrastNormalization((0.5, 1.5))),

                iaa.OneOf([
                    iaa.OneOf([
                        iaa.GaussianBlur(sigma=(0.0, 0.5)),
                        iaa.AverageBlur(k=((5, 7), (1, 3))),
                        iaa.MedianBlur(k=(3, 5))
                    ]),
                    iaa.OneOf([
                        iaa.AddElementwise((-40, 40)),
                        iaa.AddElementwise((-40, 40), per_channel=0.5)
                    ]),
                    iaa.OneOf([
                        iaa.AdditiveGaussianNoise(scale=(0, 0.005 * 255)),
                        iaa.AdditiveGaussianNoise(scale=0.005 * 255, per_channel=0.5)
                    ])
                ]),

                iaa.Sometimes(0.5,
                              iaa.OneOf([
                                  iaa.Sharpen(alpha=(0.0, 0.3), lightness=(0.5, 1.0)),
                                  iaa.Emboss(alpha=(0.0, 0.3), strength=(0.5, 1.0))
                              ])
                              ),

                iaa.Sometimes(0.5,
                              iaa.OneOf([
                                  iaa.OneOf([
                                      iaa.Add((-60, 20)),
                                      iaa.Add((-60, 20), per_channel=0.5)
                                  ]),
                                  iaa.OneOf([
                                      iaa.Multiply((0.5, 1.5)),
                                      iaa.Multiply((0.5, 1.5), per_channel=0.5)
                                  ])
                              ])
                              ),

                iaa.OneOf([
                    iaa.OneOf([
                        iaa.Dropout(p=(0, 0.2)),
                        iaa.Dropout(p=(0, 0.2), per_channel=0.5)
                    ]),
                    iaa.OneOf([
                        iaa.CoarseDropout((0.0, 0.02), size_percent=(0.02, 0.25)),
                        iaa.CoarseDropout((0.0, 0.02), size_percent=0.15, per_channel=0.5)
                    ])
                ])
            ]),
            iaa.Resize({"height": self.img_size[0], "width": self.img_size[1]})
        ], random_order=True)

        return sequential

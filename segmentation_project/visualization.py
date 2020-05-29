from segmentation_project.colors import label_color
import cv2
import numpy as np


class Visualisator:

    def get_color(self, label, color):

        if color is not None:
            return color

        if label is not None:
            color = label_color(label)

        return color


    def get_parameter(self, params, length, ind):
        if params is not None:
            assert len(params) == length, 'number of elements not equal number of bbox'
            return params[ind]
        else:
            return None


    def draw_info(self, image, info, point, color):
        x, y = point
        y += 12
        cv2.putText(image, '%s' % info, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color, lineType=cv2.LINE_AA, thickness=1)


    def draw_bbox(self, image, bbox, label=None, color=None, info=None):
        """
        :param image: RGB image
        :param bbox: list of int [x_min, y_min, x_max, y_max]
        :param label: int
        :param color: tuple of int consisting of 3 elements ranging from 0 to 255
        :param text:
        """
        x_min, y_min, x_max, y_max = bbox
        color = self.get_color(label, color)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color)

        if info is not None:
            self.draw_info(image, info, point=(x_min, y_min), color=color)


    def draw_bboxes(self, image, bboxs, labels=None, colors=None, list_of_info=None):
        length = len(bboxs)
        for i, bbox in enumerate(bboxs):
            label = self.get_parameter(labels, length, i)
            color = self.get_parameter(colors, length, i)
            info = self.get_parameter(list_of_info, length, i)
            self.draw_bbox(image, bbox, label, color, info)


    def draw_mask(self, image, mask, label=None, color=None, binarize_threshold=0.5, draw_border=False):
        color = self.get_color(label, color)
        mask = (mask > binarize_threshold).astype(np.uint8)

        # compute a nice border around the mask
        if draw_border:
            border = mask - cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=1)

        # apply color to the mask and border
        mask = (np.stack([mask] * 3, axis=2) * color).astype(np.uint8)
        if draw_border:
            border = (np.stack([border] * 3, axis=2) * (255, 255, 255)).astype(np.uint8)

        # draw the mask
        indices = np.where(mask != [0, 0, 0])
        image[indices[0], indices[1], :] = 0.5 * image[indices[0], indices[1], :] + 0.5 * mask[indices[0], indices[1], :]

        # draw the border
        if draw_border:
            indices = np.where(border != [0, 0, 0])
            image[indices[0], indices[1], :] = 0.2 * image[indices[0], indices[1], :] + 0.8 * border[indices[0], indices[1], :]

    def draw_masks(self, image, mask, label=None, color=None, binarize_threshold=0.5, draw_border=False):
        for i in range(len(label)-1):
            self.draw_mask(image, mask[:,:,i], label[i])

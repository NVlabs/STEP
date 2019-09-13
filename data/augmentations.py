"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random

from utils.tube_utils import valid_tubes

def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images, tubes=None, proposals=None):
        for t in self.transforms:
            images, tubes, proposals = t(images, tubes, proposals)
        return images, tubes, proposals

    def __str__(self):
        string = ""
        for t in self.transforms:
            string = string + t.get_name() + ' '
        return string


class ConvertFromInts(object):
    def __init__(self, scale):
        self._name = "ConvertFromInts"
        self.scale = scale
    def get_name(self):
        return self._name

    def __call__(self, images, tubes=None, proposals=None):
        if self.scale == 0:
            images = images.astype(np.float32)
        elif self.scale == 1:
            images = images.astype(np.float32) / 255.   # scale [0,255] to [0,1]
        elif self.scale == 2:
            images = np.clip(images, 0, 255)
            images = images.astype(np.float32)*2/255 - 1.   # scale [0,255] to [-1,1]

        return images, tubes, proposals


class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)
        self._name = "SubtractMeans"
    def get_name(self):
        return self._name

    def __call__(self, images, tubes=None, proposals=None):
        images = images.astype(np.float32)
        images -= self.mean
        return images.astype(np.float32), tubes, proposals

class DivideStds(object):
    def __init__(self, stds):
        self.stds = np.array(stds, dtype=np.float32)
        self._name = "DivideStds"
    def get_name(self):
        return self._name

    def __call__(self, images, tubes=None, proposals=None):
        images = images.astype(np.float32)
        images /= self.stds
        return images.astype(np.float32), tubes, proposals

class ToAbsoluteCoords(object):
    def __init__(self):
        self._name = "ToAbsoluteCoords"
    def get_name(self):
        return self._name

    def __call__(self, images, tubes=None, proposals=None):
        T, height, width, channels = images.shape
        tubes[:, :, 0] *= width
        tubes[:, :, 2] *= width
        tubes[:, :, 1] *= height
        tubes[:, :, 3] *= height

        if proposals is not None:
            proposals[:, :, 0] *= width
            proposals[:, :, 2] *= width
            proposals[:, :, 1] *= height
            proposals[:, :, 3] *= height

        return images, tubes, proposals


class ToPercentCoords(object):
    def __init__(self):
        self._name = "ToPercentCoords"
    def get_name(self):
        return self._name

    def __call__(self, images, tubes=None, proposals=None):
        T, height, width, channels = images.shape
        tubes[:,:, 0] /= width
        tubes[:,:, 2] /= width
        tubes[:,:, 1] /= height
        tubes[:,:, 3] /= height

        if proposals is not None:
            proposals[:, :, 0] /= width
            proposals[:, :, 2] /= width
            proposals[:, :, 1] /= height
            proposals[:, :, 3] /= height

        return images, tubes, proposals


class Resize(object):
    def __init__(self, size=(400,320)):
        self.size = size    # (width, height)
        self._name = "Resize"
    def get_name(self):
        return self._name

    def __call__(self, images, tubes=None, proposals=None):
        T, height, width, channels = images.shape
        resized_images = np.zeros((T, self.size[1], self.size[0], channels), dtype=images.dtype)
        for i in range(images.shape[0]):
            resized_images[i] = cv2.resize(images[i], (self.size[0],
                                 self.size[1]))
        # don't need to resize tubes because it's PercentCoords
        return resized_images, tubes, proposals


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, images, tubes=None, proposals=None):
        if random.randint(2):
            images[:, :, :, 1] *= random.uniform(self.lower, self.upper)

        return images, tubes, proposals


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, images, tubes=None, proposals=None):
        if random.randint(2):
            images[:, :, :, 0] += random.uniform(-self.delta, self.delta)
            images[:, :, :, 0][images[:, :, :, 0] > 360.0] -= 360.0
            images[:, :, :, 0][images[:, :, :, 0] < 0.0] += 360.0
        return images, tubes, proposals


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, images, tubes=None, proposals=None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            images = images[:,:,:,swap]
#            shuffle = SwapChannels(swap)  # shuffle channels
#            for i in range(images.shape[0]):
#                images[i] = shuffle(images[i])
        return images, tubes, proposals


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, images, tubes=None, proposals=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            for i in range(images.shape[0]):
                images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            for i in range(images.shape[0]):
                images[i] = cv2.cvtColor(images[i], cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return images, tubes, proposals


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, images, tubes=None, proposals=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            images *= alpha
        return images, tubes, proposals


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, images, tubes=None, proposals=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            images += delta
        return images, tubes, proposals


class RandomErase(object):
    """Random erase
    Reference: Random Erasing Data Augmentation
    """
    def __init__(self, scale=0, sl=0.02, sh=0.2, r1=0.3, r2=10/3.):
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.r2 = r2
        self.scale = scale
        self._name = "RandomErase"
    def get_name(self):
        return self._name

    def __call__(self, images, tubes=None, proposals=None):
        if random.randint(2):
            T, height, width, _ = images.shape

            # perform ORE
            boxes = tubes[:, int(tubes.shape[1]/2), :4]
            for i in range(boxes.shape[0]):
                x1,y1,x2,y2 = self.get_region(boxes[i])
                if self.scale == 0:
                    images[:,y1:y2,x1:x2] = np.random.uniform(0, 255, (y2-y1,x2-x1,3))
                elif self.scale == 1:
                    images[:,y1:y2,x1:x2] = np.random.uniform(0, 1, (y2-y1,x2-x1,3))
                elif self.scale == 2:
                    images[:,y1:y2,x1:x2] = np.random.uniform(-1, 1, (y2-y1,x2-x1,3))

        return images, tubes, proposals

    def get_region(self, region):
        x1,y1,x2,y2 = region
        S = (x2-x1) * (y2-y1)
        while True:
            Se = random.uniform(self.sl, self.sh) * S
            re = random.uniform(self.r1, self.r2)
            He = np.sqrt(Se * re)
            We = np.sqrt(Se / re)
            xe = random.uniform(x1, x2-We)
            ye = random.uniform(y1, y2-He)

            if xe + We <= x2 and ye + He <= y2:
                return [int(xe), int(ye), int(xe+We), int(ye+He)]

            


class RandomSampleCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """
    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.5, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )
        self._name = "RandomSampleCrop"
    def get_name(self):
        return self._name

    def __call__(self, images, tubes=None, proposals=None):

        T, height, width, _ = images.shape
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:
                return images, tubes, proposals

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            boxes = tubes[:, int(tubes.shape[1]/2), :4]

            # max trails (50)
            for _ in range(50):
                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h) # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(boxes, rect)

                # is min and max overlap coniou.max() <= max_ioustraint satisfied? if not try again
                if overlap.min() < min_iou or overlap.max() > max_iou:
                    continue

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                #### if all above criterions are satisfied, we accept the cropped rect ####

                # cut the crop from the image
                cropped_images = images[:, rect[1]:rect[3], rect[0]:rect[2], :]

                # take only matching gt boxes
                current_tubes = tubes[mask, :, :].copy()

                # should we use the box left and top corner or the crop's
                current_tubes[:, :, :2] = np.maximum(current_tubes[:, :, :2],
                                                  rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_tubes[:,:, :2] -= rect[:2]

                current_tubes[:,:, 2:4] = np.minimum(current_tubes[:,:, 2:4],
                                                  rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_tubes[:,:, 2:4] -= rect[:2]

                current_tubes[:,:,:4] = np.maximum(current_tubes[:,:,:4], 0.)

                if proposals is not None:
                    current_proposals = proposals.copy()
                    current_proposals[:,:,:2] = np.maximum(current_proposals[:,:,:2], rect[:2])
                    current_proposals[:,:,:2] -= rect[:2]
                    current_proposals[:,:,2:4] = np.minimum(current_proposals[:,:,2:4], rect[2:])
                    current_proposals[:,:,2:4] -= rect[:2]

                    current_proposals = valid_tubes(current_proposals, width=w, height=h)
                else:
                    current_proposals = proposals

                return cropped_images, current_tubes, current_proposals


class Expand(object):
    def __init__(self, mean):
        self.mean = mean
        self._name = "Expand"
    def get_name(self):
        return self._name

    def __call__(self, images, tubes):
        if random.randint(2):
            return images, tubes

        T, height, width, depth = images.shape
        ratio = random.uniform(1, 4)
        left = random.uniform(0, width*ratio - width)
        top = random.uniform(0, height*ratio - height)

        expand_images = np.zeros(
            (T, int(height*ratio), int(width*ratio), depth),
            dtype=images.dtype)
        expand_images[:, :, :, :] = self.mean
        expand_images[:, int(top):int(top + height),
                     int(left):int(left + width)] = images
        images = expand_images

        tubes = tubes.copy()
        tubes[:, :, :2] += (int(left), int(top))
        tubes[:, :, 2:4] += (int(left), int(top))

        return images, tubes


class RandomMirror(object):
    def __init__(self):
        self._name = "RandomMirror"
    def get_name(self):
        return self._name

    def __call__(self, images, tubes, proposals):

        T, _, width, _ = images.shape
        new_tubes = tubes.copy()
        if proposals is not None:
            new_proposals = proposals.copy()
        else:
            new_proposals = None

        if random.randint(2):
            images = images[:, :, ::-1]
            for i in range(tubes.shape[0]):
                for j in range(tubes.shape[1]):
                    if np.sum(tubes[i,j,:4]) > 0:
                        new_tubes[i, j, 0] = width - tubes[i, j, 2]
                        new_tubes[i, j, 2] = width - tubes[i, j, 0]

            if proposals is not None:
                new_proposals[:,:,0] = width - proposals[:,:,2]
                new_proposals[:,:,2] = width - proposals[:,:,0]
        return images, new_tubes, new_proposals


class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()
        self._name = "PhotometricDistort"
    def get_name(self):
        return self._name

    def __call__(self, images, tubes, proposals=None):

        ims = images.copy()
        ims, tubes, proposals = self.rand_brightness(ims, tubes, proposals)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        ims, tubes, proposals = distort(ims, tubes, proposals)
        return self.rand_light_noise(ims, tubes, proposals)


class TubeAugmentation(object):
    def __init__(self, size=300, mean=(0,0,0), stds=(1,1,1), do_flip=False, do_crop=False, do_photometric=False, do_erase=False, scale=1):
        self.mean = mean
        self.stds = stds
        self.size = size
        self.do_flip = do_flip
        self.do_crop = do_crop
        self.do_photometric = do_photometric
        self.do_erase = do_erase
        self.scale = scale

        aug_list = []

        if self.do_photometric:
            aug_list += [
                ConvertFromInts(scale=0),
                PhotometricDistort()
                ]

        aug_list += [
            ConvertFromInts(self.scale),
            ToAbsoluteCoords()
            ]

        if self.do_crop:
            aug_list += [ RandomSampleCrop() ]

        if self.do_flip:
            aug_list += [ RandomMirror() ]

        if self.do_erase:
            aug_list += [ RandomErase(scale=self.scale) ]

        aug_list += [
            ToPercentCoords(),
            Resize(self.size),
            SubtractMeans(self.mean),
            DivideStds(self.stds)
            ]

        self.augment = Compose(aug_list)

    def __call__(self, images, tubes, proposals=None):
        """
        perform consistent augmentation on a sequence of images/boxes
        """
        return self.augment(images, tubes, proposals)
    
    def __str__(self):
        return str(self.augment) + '\n'

def base_transform(images, size, mean):
    T, w, h, c = images.shape
    resized_images = np.zeros((T, size, size, c), dtype=images.dtype)
    for i in range(T):
        resized_images[i] = cv2.resize(images[i], (size, size)).astype(np.float32)
    resized_images = resized_images.astype(np.float32)
    resized_images -= mean
    return resized_images.astype(np.float32)


class BaseTransform(object):
    def __init__(self, size=(400,320), mean=(0, 0, 0), stds=(1,1,1), scale=1):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)
        self.stds = np.array(stds, dtype=np.float32)
        self.scale = scale    # 0: [0, 255], 1: [0,1], 2: [-1,1]
        self.augment = Compose([
            ConvertFromInts(self.scale),
            Resize(self.size),
            SubtractMeans(self.mean),
            DivideStds(self.stds)
        ])

    def __call__(self, images, tubes=None, proposals=None):
        return self.augment(images, tubes, proposals)

    def __str__(self):
        return str(self.augment) + '\n'

# -*- coding:utf-8 -*-
###
# File: data_process.py
# Created Date: Friday, September 27th 2019, 11:00:00 pm
# Author: yusnows
# -----
# Last Modified:
# Modified By:
# -----
# Copyright (c) 2019 yusnows
#
# All shall be well and all shall be well and all manner of things shall be well.
# Nope...we're doomed!
# -----
# HISTORY:
# Date      	By	Comments
# ----------	---	----------------------------------------------------------
###
import random
import torch
import torchvision.transforms as transforms
import PIL.Image as Image
import numpy as np
import PIL.ImageFilter as ImageFilter
import PIL.ImageEnhance as ImageEnhance
import PIL.ImageOps as ImageOps
import math
import warnings

__all__ = ["_imagenet_pca", "Lighting", "ExifTranspose", "HistogramEqualisation", "ResizeFill",
           "RandomCrop", "RandomBrightness",
           "RandomHSVShift", "RandomBlur", "RandomNoise", "RandomAffine", "RandomColor",
           "RandomContrast", "RandomFlip", "RandomShear", "RandomErasing", "RandomZoom",
           "LabelSmoothOnehot", "ERandomCrop", "ECenterCrop"]


_imagenet_pca = {
    'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': torch.Tensor([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ])
}


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


class ExifTranspose(object):
    def __init__(self):
        super(ExifTranspose, self).__init__()

    def __call__(self, img):
        if not img:
            return img
        exif_orientation_tag = 274
        # Check for EXIF data (only present on some files)
        if hasattr(img, "_getexif") and isinstance(img._getexif(), dict) and exif_orientation_tag in img._getexif():
            exif_data = img._getexif()
            orientation = exif_data[exif_orientation_tag]
            # Handle EXIF Orientation
            if orientation == 1:
                # Normal image - nothing to do!
                pass
            elif orientation == 2:
                # Mirrored left to right
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            elif orientation == 3:
                # Rotated 180 degrees
                img = img.rotate(180)
            elif orientation == 4:
                # Mirrored top to bottom
                img = img.rotate(180).transpose(Image.FLIP_LEFT_RIGHT)
            elif orientation == 5:
                # Mirrored along top-left diagonal
                img = img.rotate(-90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
            elif orientation == 6:
                # Rotated 90 degrees
                img = img.rotate(-90, expand=True)
            elif orientation == 7:
                # Mirrored along top-right diagonal
                img = img.rotate(90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
            elif orientation == 8:
                # Rotated 270 degrees
                img = img.rotate(90, expand=True)
        return img


class HistogramEqualisation(object):
    """
    The class :class:`HistogramEqualisation` is used to perform histogram
    equalisation on images passed to its :func:`perform_operation` function.
    """

    def __init__(self, p=1):
        """
        As there are no further user definable parameters, the class is
        instantiated using only the :attr:`probability` argument.
        :param probability: Controls the probability that the operation is
         performed when it is invoked in the pipeline.
        :type probability: Float
        """
        super(HistogramEqualisation, self).__init__()
        self.probability = p

    def __call__(self, img):
        """
        Performs histogram equalisation on the images passed as an argument
        and returns the equalised images. There are no user definable
        parameters for this method.
        :param images: The image(s) on which to perform the histogram
         equalisation.
        :type images: List containing PIL.Image object(s).
        :return: The transformed image(s) as a list of object(s) of type
         PIL.Image.
        """
        # If an image is a colour image, the histogram will
        # will be computed on the flattened image, which fires
        # a warning.
        # We may want to apply this instead to each colour channel.
        if np.random.random() < self.probability:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                img = ImageOps.equalize(img)
        return img


class ResizeFill(object):
    """
    resize the images
    size: int or (int, int) for (height, width)
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        super(ResizeFill, self).__init__()
        assert isinstance(size, (int, tuple))
        if isinstance(size, int):
            self.size = (size, size)
        else:
            assert len(size) == 2
            self.size = (size[1], size[0])  # self.size is (w, h)
        self.interpolation = interpolation

    def __call__(self, img):
        if (self.size[0] == img.size[0]) and (self.size[1] == img.size[1]):
            return img
        ratio = self.size[0] / self.size[1]
        w, h = img.size
        if w/h < ratio:
            t = int(h*ratio)
            w_padding = (t-w)//2
            img = img.crop((-w_padding, 0, w+w_padding, h))
        else:
            t = int(w/ratio)
            h_padding = (t-h)//2
            img = img.crop((0, -h_padding, w, h+h_padding))
        img = img.resize(self.size, self.interpolation)
        return img


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, p=0.9, scale=(0.1, 1.0)):
        super(RandomCrop, self).__init__()
        self.scale = scale
        self.probability = p

    def __call__(self, img):
        if np.random.random() < self.probability:
            scale = np.random.uniform(self.scale[0], self.scale[1])
            w, h = img.size
            new_h, new_w = int(h*scale), int(w*scale)
            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)
            img = img.crop((left, top, left + new_w, top + new_h))
        return img


class RandomHSVShift(object):
    def __init__(self, p=0.9):
        super(RandomHSVShift, self).__init__()
        self.probability = p

    def __call__(self, img):
        prob = np.random.random()
        if prob <= self.probability:
            fraction = 0.50
            img_hsv = img.convert('HSV')
            img_hsv = np.array(img_hsv)
            # 类似于BGR，HSV的shape=(w,h,c)，其中三通道的c[0,1,2]含有h,s,v信息
            S = img_hsv[:, :, 1].astype(np.float32)
            V = img_hsv[:, :, 2].astype(np.float32)
            a = (np.random.random() * 2 - 1) * fraction + 1
            S *= a
            if a > 1:
                np.clip(S, a_min=0, a_max=255, out=S)
            a = (np.random.random() * 2 - 1) * fraction + 1
            V *= a
            if a > 1:
                np.clip(V, a_min=0, a_max=255, out=V)
            img_hsv[:, :, 1] = S.astype(np.uint8)
            img_hsv[:, :, 2] = V.astype(np.uint8)
            img_hsv = Image.fromarray(img_hsv, 'HSV')
            img = img_hsv.convert("RGB")
        return img


class RandomBlur(object):
    def __init__(self, p=0.5):
        super(RandomBlur, self).__init__()
        self.probability = p

    def __call__(self, img):
        prob = np.random.random()
        if prob < self.probability:
            img = img.filter(ImageFilter.BLUR)
        return img


class RandomNoise(object):
    def __init__(self, p=0.5):
        super(RandomNoise, self).__init__()
        self.probability = p

    def __call__(self, img):
        prob = np.random.random()
        if prob < self.probability:
            img = np.array(img, dtype=np.int32)
            noise = np.random.normal(0, 10, img.shape)
            img = img + noise.astype(np.int32)
            np.clip(img, a_min=0, a_max=255, out=img)
            img = img.astype(np.uint8)
            img = Image.fromarray(img, 'RGB')
        return img


class RandomAffine(object):
    def __init__(self, degrees, p=0.5, translate=None, scale=None, shear=None, resample=False, fillcolor=0):
        super(RandomAffine, self).__init__()
        self.probability = p
        self.trans = transforms.RandomAffine(degrees, translate, scale, shear, resample, fillcolor)

    def __call__(self, img):
        if np.random.random() < self.probability:
            img = self.trans(img)
        return img


class RandomBrightness(object):
    """
    This class is used to random change image brightness.
    """

    def __init__(self, p=0.8, min_factor=0.4, max_factor=1.33):
        """
        required :attr:`probability` parameter
        :func:`~Augmentor.Pipeline.Pipeline.random_brightness` function.
        :param probability: Controls the probability that the operation is
         performed when it is invoked in the pipeline.
        :param min_factor: The value between 0.0 and max_factor
         that define the minimum adjustment of image brightness.
         The value  0.0 gives a black image,The value 1.0 gives the original image, value bigger than 1.0 gives more bright image.
        :param max_factor: A value should be bigger than min_factor.
         that define the maximum adjustment of image brightness.
         The value  0.0 gives a black image, value 1.0 gives the original image, value bigger than 1.0 gives more bright image.
        :type probability: Float
        :type max_factor: Float
        :type max_factor: Float
        """
        super(RandomBrightness, self).__init__()
        self.probability = p
        self.min_factor = min_factor
        self.max_factor = max_factor

    def __call__(self, img):
        """
        Random change the passed image brightness.
        :param images: The image to convert into monochrome.
        :type images: List containing PIL.Image object(s).
        :return: The transformed image(s) as a list of object(s) of type
         PIL.Image.
        """
        if np.random.random() < self.probability:
            factor = np.random.uniform(self.min_factor, self.max_factor)
            image_enhancer_brightness = ImageEnhance.Brightness(img)
            img = image_enhancer_brightness.enhance(factor)
        return img


class RandomColor(object):
    """
    This class is used to random change saturation of an image.
    """

    def __init__(self, p=0.9, min_factor=0.4, max_factor=1.8):
        """
        required :attr:`probability` parameter
        :func:`~Augmentor.Pipeline.Pipeline.random_color` function.
        :param probability: Controls the probability that the operation is
         performed when it is invoked in the pipeline.
        :param min_factor: The value between 0.0 and max_factor
         that define the minimum adjustment of image saturation.
         The value 0.0 gives a black and white image, value 1.0 gives the original image.
        :param max_factor: A value should be bigger than min_factor.
         that define the maximum adjustment of image saturation.
         The value 0.0 gives a black and white image, value 1.0 gives the original image.
        """
        super(RandomColor, self).__init__()
        self.probability = p
        self.min_factor = min_factor
        self.max_factor = max_factor

    def __call__(self, image):
        """
        Random change the passed image saturation.
        :param images: The image to convert into monochrome.
        :type images: List containing PIL.Image object(s).
        :return: The transformed image(s) as a list of object(s) of type
         PIL.Image.
        """
        if np.random.random() < self.probability:
            factor = np.random.uniform(self.min_factor, self.max_factor)
            image_enhancer_color = ImageEnhance.Color(image)
            image = image_enhancer_color.enhance(factor)
        return image


class RandomContrast(object):
    """
    This class is used to random change contrast of an image.
    """

    def __init__(self, p=0.9, min_factor=0.4, max_factor=1.8):
        """
        required :attr:`probability` parameter
        :func:`~Augmentor.Pipeline.Pipeline.random_contrast` function.
        :param probability: Controls the probability that the operation is
         performed when it is invoked in the pipeline.
        :param min_factor: The value between 0.0 and max_factor
         that define the minimum adjustment of image contrast.
         The value  0.0 gives s solid grey image, value 1.0 gives the original image.
        :param max_factor: A value should be bigger than min_factor.
         that define the maximum adjustment of image contrast.
         The value  0.0 gives s solid grey image, value 1.0 gives the original image.
        :type probability: Float
        :type max_factor: Float
        :type max_factor: Float
        """
        self.probability = p
        self.min_factor = min_factor
        self.max_factor = max_factor

    def __call__(self, image):
        """
        Random change the passed image contrast.
        :param images: The image to convert into monochrome.
        :type images: List containing PIL.Image object(s).
        :return: The transformed image(s) as a list of object(s) of type
         PIL.Image.
        """
        if np.random.random() < self.probability:
            factor = np.random.uniform(self.min_factor, self.max_factor)
            # factor = 2
            image_enhancer_contrast = ImageEnhance.Contrast(image)
            image = image_enhancer_contrast.enhance(factor)
        return image


class RandomSkew(object):
    """
    This class is used to perform perspective skewing on images. It allows
    for skewing from a total of 12 different perspectives.
    """

    def __init__(self, p=0.9, skew_type='RANDOM', magnitude=10):
        """
        As well as the required :attr:`probability` parameter, the type of
        skew that is performed is controlled using a :attr:`skew_type` and a
        :attr:`magnitude` parameter. The :attr:`skew_type` controls the
        direction of the skew, while :attr:`magnitude` controls the degree
        to which the skew is performed.
        To see examples of the various skews, see :ref:`perspectiveskewing`.
        Images are skewed **in place** and an image of the same size is
        returned by this function. That is to say, that after a skew
        has been performed, the largest possible area of the same aspect ratio
        of the original image is cropped from the skewed image, and this is
        then resized to match the original image size. The
        :ref:`perspectiveskewing` section describes this in detail.
        :param probability: Controls the probability that the operation is
         performed when it is invoked in the pipeline.
        :param skew_type: Must be one of ``TILT``, ``TILT_TOP_BOTTOM``,
         ``TILT_LEFT_RIGHT``, or ``CORNER``.
         - ``TILT`` will randomly skew either left, right, up, or down.
           Left or right means it skews on the x-axis while up and down
           means that it skews on the y-axis.
         - ``TILT_TOP_BOTTOM`` will randomly skew up or down, or in other
           words skew along the y-axis.
         - ``TILT_LEFT_RIGHT`` will randomly skew left or right, or in other
           words skew along the x-axis.
         - ``CORNER`` will randomly skew one **corner** of the image either
           along the x-axis or y-axis. This means in one of 8 different
           directions, randomly.
         To see examples of the various skews, see :ref:`perspectiveskewing`.
        :param magnitude: The degree to which the image is skewed.
        :type probability: Float
        :type skew_type: String
        :type magnitude: Integer
        """
        super(RandomSkew, self).__init__()
        self.probability = p
        self.skew_type = skew_type
        self.magnitude = magnitude

    def __call__(self, image):
        """
        Perform the skew on the passed image(s) and returns the transformed
        image(s). Uses the :attr:`skew_type` and :attr:`magnitude` parameters
        to control the type of skew to perform as well as the degree to which
        it is performed.
        If a list of images is passed, they must have identical dimensions.
        This is checked when we add the ground truth directory using
        :func:`Pipeline.:func:`~Augmentor.Pipeline.Pipeline.ground_truth`
        function.
        However, if this check fails, the skew function will be skipped and
        a warning thrown, in order to avoid an exception.
        :param images: The image(s) to skew.
        :type images: List containing PIL.Image object(s).
        :return: The transformed image(s) as a list of object(s) of type
         PIL.Image.
        """
        if np.random.random() < self.probability:
            # Width and height taken from first image in list.
            # This requires that all ground truth images in the list
            # have identical dimensions!
            w, h = image.size
            x1 = 0
            x2 = h
            y1 = 0
            y2 = w
            original_plane = [(y1, x1), (y2, x1), (y2, x2), (y1, x2)]
            max_skew_amount = max(w, h)
            max_skew_amount = int(np.ceil(max_skew_amount * self.magnitude))
            skew_amount = np.random.randint(1, max_skew_amount)
            if self.skew_type == "RANDOM":
                skew = np.random.choice(["TILT", "TILT_LEFT_RIGHT", "TILT_TOP_BOTTOM", "CORNER"])
            else:
                skew = self.skew_type
            # We have two choices now: we tilt in one of four directions
            # or we skew a corner.
            if skew == "TILT" or skew == "TILT_LEFT_RIGHT" or skew == "TILT_TOP_BOTTOM":
                if skew == "TILT":
                    skew_direction = np.random.randint(0, 3)
                elif skew == "TILT_LEFT_RIGHT":
                    skew_direction = np.random.randint(0, 1)
                elif skew == "TILT_TOP_BOTTOM":
                    skew_direction = np.random.randint(2, 3)
                else:
                    skew_direction = np.random.randint(0, 3)

                if skew_direction == 0:
                    # Left Tilt
                    new_plane = [(y1, x1 - skew_amount),  # Top Left
                                 (y2, x1),                # Top Right
                                 (y2, x2),                # Bottom Right
                                 (y1, x2 + skew_amount)]  # Bottom Left
                elif skew_direction == 1:
                    # Right Tilt
                    new_plane = [(y1, x1),                # Top Left
                                 (y2, x1 - skew_amount),  # Top Right
                                 (y2, x2 + skew_amount),  # Bottom Right
                                 (y1, x2)]                # Bottom Left
                elif skew_direction == 2:
                    # Forward Tilt
                    new_plane = [(y1 - skew_amount, x1),  # Top Left
                                 (y2 + skew_amount, x1),  # Top Right
                                 (y2, x2),                # Bottom Right
                                 (y1, x2)]                # Bottom Left
                elif skew_direction == 3:
                    # Backward Tilt
                    new_plane = [(y1, x1),                # Top Left
                                 (y2, x1),                # Top Right
                                 (y2 + skew_amount, x2),  # Bottom Right
                                 (y1 - skew_amount, x2)]  # Bottom Left
            if skew == "CORNER":
                skew_direction = np.random.randint(0, 7)
                if skew_direction == 0:
                    # Skew possibility 0
                    new_plane = [(y1 - skew_amount, x1), (y2, x1), (y2, x2), (y1, x2)]
                elif skew_direction == 1:
                    # Skew possibility 1
                    new_plane = [(y1, x1 - skew_amount), (y2, x1), (y2, x2), (y1, x2)]
                elif skew_direction == 2:
                    # Skew possibility 2
                    new_plane = [(y1, x1), (y2 + skew_amount, x1), (y2, x2), (y1, x2)]
                elif skew_direction == 3:
                    # Skew possibility 3
                    new_plane = [(y1, x1), (y2, x1 - skew_amount), (y2, x2), (y1, x2)]
                elif skew_direction == 4:
                    # Skew possibility 4
                    new_plane = [(y1, x1), (y2, x1), (y2 + skew_amount, x2), (y1, x2)]
                elif skew_direction == 5:
                    # Skew possibility 5
                    new_plane = [(y1, x1), (y2, x1), (y2, x2 + skew_amount), (y1, x2)]
                elif skew_direction == 6:
                    # Skew possibility 6
                    new_plane = [(y1, x1), (y2, x1), (y2, x2), (y1 - skew_amount, x2)]
                elif skew_direction == 7:
                    # Skew possibility 7
                    new_plane = [(y1, x1), (y2, x1), (y2, x2), (y1, x2 + skew_amount)]
            if self.skew_type == "ALL":
                # Not currently in use, as it makes little sense to skew by the same amount
                # in every direction if we have set magnitude manually.
                # It may make sense to keep this, if we ensure the skew_amount below is randomised
                # and cannot be manually set by the user.
                corners = dict()
                corners["top_left"] = (y1 - np.random.randint(1, skew_amount), x1 - np.random.randint(1, skew_amount))
                corners["top_right"] = (y2 + np.random.randint(1, skew_amount), x1 - np.random.randint(1, skew_amount))
                corners["bottom_right"] = (y2 + np.random.randint(1, skew_amount),
                                           x2 + np.random.randint(1, skew_amount))
                corners["bottom_left"] = (y1 - np.random.randint(1, skew_amount),
                                          x2 + np.random.randint(1, skew_amount))
                new_plane = [corners["top_left"], corners["top_right"], corners["bottom_right"], corners["bottom_left"]]

            matrix = []
            for p1, p2 in zip(new_plane, original_plane):
                matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
                matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])
            A = np.matrix(matrix, dtype=np.float)
            B = np.array(original_plane).reshape(8)
            perspective_skew_coefficients_matrix = np.dot(np.linalg.pinv(A), B)
            perspective_skew_coefficients_matrix = np.array(perspective_skew_coefficients_matrix).reshape(8)
            image = image.transform(image.size, Image.PERSPECTIVE,
                                    perspective_skew_coefficients_matrix, resample=Image.BICUBIC)
        return image


class RandomFlip(object):
    """
    This class is used to mirror images through the x or y axes.
    The class allows an image to be mirrored along either
    its x axis or its y axis, or randomly.
    """

    def __init__(self, p=0.5, top_bottom_left_right='LEFT_RIGHT'):
        """
        The direction of the flip, or whether it should be randomised, is
        controlled using the :attr:`top_bottom_left_right` parameter.
        :param probability: Controls the probability that the operation is
         performed when it is invoked in the pipeline.
        :param top_bottom_left_right: Controls the direction the image should
         be mirrored. Must be one of ``LEFT_RIGHT``, ``TOP_BOTTOM``, or
         ``RANDOM``.
         - ``LEFT_RIGHT`` defines that the image is mirrored along its x axis.
         - ``TOP_BOTTOM`` defines that the image is mirrored along its y axis.
         - ``RANDOM`` defines that the image is mirrored randomly along
           either the x or y axis.
        """
        super(RandomFlip, self).__init__()
        self.probability = p
        self.top_bottom_left_right = top_bottom_left_right

    def __call__(self, image):
        """
        Mirror the image according to the `attr`:top_bottom_left_right`
        argument passed to the constructor and return the mirrored image.
        :param images: The image(s) to mirror.
        :type images: List containing PIL.Image object(s).
        :return: The transformed image(s) as a list of object(s) of type
         PIL.Image.
        """
        if np.random.random() < self.probability:
            random_axis = np.random.randint(0, 1)
            if self.top_bottom_left_right == "LEFT_RIGHT":
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            elif self.top_bottom_left_right == "TOP_BOTTOM":
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
            elif self.top_bottom_left_right == "RANDOM":
                if random_axis == 0:
                    image = image.transpose(Image.FLIP_LEFT_RIGHT)
                elif random_axis == 1:
                    image = image.transpose(Image.FLIP_TOP_BOTTOM)
        return image


class RandomShear(object):
    """
    This class is used to shear images, that is to tilt them in a certain
    direction. Tilting can occur along either the x- or y-axis and in both
    directions (i.e. left or right along the x-axis, up or down along the
    y-axis).
    Images are sheared **in place** and an image of the same size as the input
    image is returned by this class. That is to say, that after a shear
    has been performed, the largest possible area of the same aspect ratio
    of the original image is cropped from the sheared image, and this is
    then resized to match the original image size. The
    :ref:`shearing` section describes this in detail.
    For sample code with image examples see :ref:`shearing`.
    """

    def __init__(self, p=0.8, max_shear_left=20, max_shear_right=20, axis=None):
        """
        The shearing is randomised in magnitude, from 0 to the
        :attr:`max_shear_left` or 0 to :attr:`max_shear_right` where the
        direction is randomised. The shear axis is also randomised
        i.e. if it shears up/down along the y-axis or
        left/right along the x-axis.
        :param probability: Controls the probability that the operation is
         performed when it is invoked in the pipeline.
        :param max_shear_left: The maximum shear to the left.
        :param max_shear_right: The maximum shear to the right.
        :type probability: Float
        :type max_shear_left: Integer
        :type max_shear_right: Integer
        """
        super(RandomShear, self).__init__()
        self.probability = p
        self.max_shear_left = max_shear_left
        self.max_shear_right = max_shear_right
        self.axis = axis

    def __call__(self, image):
        """
        Shears the passed image according to the parameters defined during
        instantiation, and returns the sheared image.
        :param images: The image to shear.
        :type images: List containing PIL.Image object(s).
        :return: The transformed image(s) as a list of object(s) of type
         PIL.Image.
        """
        if np.random.random() < self.probability:
            width, height = image.size
            angle_to_shear = int(np.random.uniform((abs(self.max_shear_left)*-1) - 1, self.max_shear_right + 1))
            if angle_to_shear != -1:
                angle_to_shear += 1
            if self.axis is None:
                directions = ["x", "y"]
            else:
                directions = self.axis
            direction = np.random.choice(directions)
            # We use the angle phi in radians later
            phi = math.tan(math.radians(angle_to_shear))
            if direction == "x":
                # Here we need the unknown b, where a is
                # the height of the image and phi is the
                # angle we want to shear (our knowns):
                # b = tan(phi) * a
                shift_in_pixels = phi * height
                if shift_in_pixels > 0:
                    shift_in_pixels = math.ceil(shift_in_pixels)
                else:
                    shift_in_pixels = math.floor(shift_in_pixels)
                # For negative tilts, we reverse phi and set offset to 0
                # Also matrix offset differs from pixel shift for neg
                # but not for pos so we will copy this value in case
                # we need to change it
                matrix_offset = shift_in_pixels
                if angle_to_shear <= 0:
                    shift_in_pixels = abs(shift_in_pixels)
                    matrix_offset = 0
                    phi = abs(phi) * -1
                # Note: PIL expects the inverse scale, so 1/scale_factor for example.
                transform_matrix = (1, phi, -matrix_offset,
                                    0, 1, 0)
                image = image.transform((int(round(width + shift_in_pixels)), height),
                                        Image.AFFINE,
                                        transform_matrix,
                                        Image.BICUBIC)
                image = image.crop((abs(shift_in_pixels), 0, width, height))

                return image.resize((width, height), resample=Image.BICUBIC)
            elif direction == "y":
                shift_in_pixels = phi * width
                matrix_offset = shift_in_pixels
                if angle_to_shear <= 0:
                    shift_in_pixels = abs(shift_in_pixels)
                    matrix_offset = 0
                    phi = abs(phi) * -1
                transform_matrix = (1, 0, 0,
                                    phi, 1, -matrix_offset)
                image = image.transform((width, int(round(height + shift_in_pixels))),
                                        Image.AFFINE,
                                        transform_matrix,
                                        Image.BICUBIC)
                image = image.crop((0, abs(shift_in_pixels), width, height))
                return image.resize((width, height), resample=Image.BICUBIC)
        return image


class RandomErasing(object):
    """
    Class that performs Random Erasing, an augmentation technique described
    in `https://arxiv.org/abs/1708.04896 <https://arxiv.org/abs/1708.04896>`_
    by Zhong et al. To quote the authors, random erasing:
    "*... randomly selects a rectangle region in an image, and erases its
    pixels with random values.*"
    Exactly this is provided by this class.
    Random Erasing can make a trained neural network more robust to occlusion.
    """

    def __init__(self, p=0.5, rectangle_area=0.2):
        """
        The size of the random rectangle is controlled using the
        :attr:`rectangle_area` parameter. This area is random in its
        width and height.
        :param probability: The probability that the operation will be
         performed.
        :param rectangle_area: The percentage are of the image to occlude.
        """
        super(RandomErasing, self).__init__()
        self.probability = p
        self.rectangle_area = rectangle_area

    def __call__(self, image):
        """
        Adds a random noise rectangle to a random area of the passed image,
        returning the original image with this rectangle superimposed.
        :param images: The image(s) to add a random noise rectangle to.
        :type images: List containing PIL.Image object(s).
        :return: The transformed image(s) as a list of object(s) of type
         PIL.Image.
        """

        def do(image):
            w, h = image.size
            w_occlusion_max = int(w * self.rectangle_area)
            h_occlusion_max = int(h * self.rectangle_area)
            w_occlusion_min = int(w * 0.1)
            h_occlusion_min = int(h * 0.1)
            w_occlusion = np.random.randint(w_occlusion_min, w_occlusion_max)
            h_occlusion = np.random.randint(h_occlusion_min, h_occlusion_max)
            if len(image.getbands()) == 1:
                rectangle = Image.fromarray(np.uint8(np.random.rand(w_occlusion, h_occlusion) * 255))
            else:
                rectangle = Image.fromarray(np.uint8(np.random.rand(
                    w_occlusion, h_occlusion, len(image.getbands())) * 255))
            random_position_x = np.random.randint(0, w - w_occlusion)
            random_position_y = np.random.randint(0, h - h_occlusion)
            image.paste(rectangle, (random_position_x, random_position_y))
            return image

        if np.random.random() < self.probability:
            image = do(image)
        return image


class RandomZoom(object):
    """
    This class is used to zoom into random areas of the image.
    """

    def __init__(self, p=1, percentage_area=(0.7, 1.0)):
        """
        Zooms into a random area of the image, rather than the centre of
        the image, as is done by :class:`Zoom`. The zoom factor is fixed
        unless :attr:`randomise` is set to ``True``.
        :param probability: Controls the probability that the operation is
         performed when it is invoked in the pipeline.
        :param percentage_area: A value between 0.1 and 1 that represents the
         area that will be cropped, with 1 meaning the entire area of the
         image will be cropped and 0.1 mean 10% of the area of the image
         will be cropped, before zooming.
        :param randomise: If ``True``, uses the :attr:`percentage_area` as an
         upper bound, and randomises the zoom level from between 0.1 and
         :attr:`percentage_area`.
        """
        super(RandomZoom, self).__init__()
        self.probability = p
        self.percentage_area = percentage_area

    def __call__(self, img):
        """
        Randomly zoom into the passed :attr:`images` by first cropping the image
        based on the :attr:`percentage_area` argument, and then resizing the
        image to match the size of the input area.
        Effectively, you are zooming in on random areas of the image.
        :param images: The image to crop an area from.
        :type images: List containing PIL.Image object(s).
        :return: The transformed image(s) as a list of object(s) of type
         PIL.Image.
        """
        if np.random.random() < self.probability:
            r_percentage_area = np.random.uniform(self.percentage_area[0], self.percentage_area[1])
            w, h = img.size
            w_new = int(np.floor(w * r_percentage_area))
            h_new = int(np.floor(h * r_percentage_area))
            random_left_shift = np.random.randint(0, (w - w_new))  # Note: randint() is from uniform distribution.
            random_down_shift = np.random.randint(0, (h - h_new))
            img = img.crop((random_left_shift, random_down_shift,
                            w_new + random_left_shift, h_new + random_down_shift))
            img = img.resize((w, h), resample=Image.BICUBIC)
        return img


class LabelSmoothOnehot(object):
    def __init__(self, classes: int, smoothing=0.0):
        super(LabelSmoothOnehot, self).__init__()
        self.classes = classes
        self.smoothing = smoothing

    def __call__(self, target):
        assert 0 <= self.smoothing < 1
        confidence = 1.0 - self.smoothing
        label_shape = torch.Size((target.size(0), self.classes))
        with torch.no_grad():
            true_dist = torch.empty(size=label_shape, device=target.device)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), confidence)
        return true_dist


# https://github.com/kakaobrain/fast-autoaugment/blob/master/FastAutoAugment/data.py
class ERandomCrop:
    def __init__(self, imgsize, min_covered=0.1, aspect_ratio_range=(3./4, 4./3),
                 area_range=(0.1, 1.0), max_attempts=10):
        assert 0.0 < min_covered
        assert 0 < aspect_ratio_range[0] <= aspect_ratio_range[1]
        assert 0 < area_range[0] <= area_range[1]
        assert 1 <= max_attempts

        self.imgsize = imgsize
        self.min_covered = min_covered
        self.aspect_ratio_range = aspect_ratio_range
        self.area_range = area_range
        self.max_attempts = max_attempts
        self._fallback = ECenterCrop(imgsize)
        self.resize_method = transforms.Resize((imgsize, imgsize), interpolation=Image.BICUBIC)

    def __call__(self, img):
        original_width, original_height = img.size
        min_area = self.area_range[0] * (original_width * original_height)
        max_area = self.area_range[1] * (original_width * original_height)

        for _ in range(self.max_attempts):
            aspect_ratio = random.uniform(*self.aspect_ratio_range)
            height = int(round(math.sqrt(min_area / aspect_ratio)))
            max_height = int(round(math.sqrt(max_area / aspect_ratio)))

            if max_height * aspect_ratio > original_width:
                max_height = (original_width + 0.5 - 1e-7) / aspect_ratio
                max_height = int(max_height)
                if max_height * aspect_ratio > original_width:
                    max_height -= 1

            if max_height > original_height:
                max_height = original_height

            if height >= max_height:
                height = max_height

            height = int(round(random.uniform(height, max_height)))
            width = int(round(height * aspect_ratio))
            area = width * height

            if area < min_area or area > max_area:
                continue
            if width > original_width or height > original_height:
                continue
            if area < self.min_covered * (original_width * original_height):
                continue
            if width == original_width and height == original_height:
                return self._fallback(img)

            x = random.randint(0, original_width - width)
            y = random.randint(0, original_height - height)
            img = img.crop((x, y, x + width, y + height))
            return self.resize_method(img)

        return self._fallback(img)


class ECenterCrop:
    """Crop the given PIL Image and resize it to desired size.
    Args:
        img (PIL Image): Image to be cropped. (0,0) denotes the top left corner of the image.
        output_size (sequence or int): (height, width) of the crop box. If int,
            it is used for both directions
    Returns:
        PIL Image: Cropped image.
    """

    def __init__(self, imgsize):
        self.imgsize = imgsize
        self.resize_method = transforms.Resize((imgsize, imgsize), interpolation=Image.BICUBIC)

    def __call__(self, img):
        image_width, image_height = img.size
        image_short = min(image_width, image_height)

        crop_size = float(self.imgsize) / (self.imgsize + 32) * image_short

        crop_height, crop_width = crop_size, crop_size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        img = img.crop((crop_left, crop_top, crop_left + crop_width, crop_top + crop_height))
        return self.resize_method(img)


if __name__ == "__main__":
    im_file = "/home/yusnows/Documents/DataSets/competition/weatherRecog/original/test.jpg"
    im_save = "/home/yusnows/Documents/DataSets/competition/weatherRecog/original/test-1.jpg"
    img = Image.open(im_file)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    trans = transforms.Compose([
        # RandomAffine(30, p=0.9, translate=(0.1, 0.1)),
        # RandomCrop(scale=(0.8, 1.0)),
        ResizeFill(280),
        # RandomBlur(p=0.9),
        # RandomNoise(p=0.75),
        # RandomErasing(p=0.9),
        # RandomShear(),
        # RandomSkew(p=0.9, magnitude=1),
        # RandomContrast(p=1),
        # RandomColor(p=0.9),
        # RandomHSVShift(p=1),
        # RandomFlip(p=0.9),
        # RandomZoom(),
        RandomBrightness(p=0.9),
    ])
    img = trans(img)
    img.save(im_save)

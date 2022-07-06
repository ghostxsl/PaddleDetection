# Modified by PaddleDetection
# 2022.08.09
#

# Copyright 2019 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import glob
import random
import logging
import cv2

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda

import tensorrt as trt

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def get_calibration_files(calibration_data,
                          max_calibration_size=None,
                          allowed_extensions=(".jpeg", ".jpg", ".png")):
    """Returns a list of all filenames ending with `allowed_extensions` found in the `calibration_data` directory.

    Parameters
    ----------
    calibration_data: str
        Path to directory containing desired files.
    max_calibration_size: int
        Max number of files to use for calibration. If calibration_data contains more than this number,
        a random sample of size max_calibration_size will be returned instead. If None, all samples will be used.

    Returns
    -------
    calibration_files: List[str]
         List of filenames contained in the `calibration_data` directory ending with `allowed_extensions`.
    """

    logger.info("Collecting calibration files from: {:}".format(
        calibration_data))
    calibration_files = [
        path
        for path in glob.iglob(
            os.path.join(calibration_data, "**"), recursive=True)
        if os.path.isfile(path) and path.lower().endswith(allowed_extensions)
    ]
    logger.info("Number of Calibration Files found: {:}".format(
        len(calibration_files)))

    if len(calibration_files) == 0:
        raise Exception("ERROR: Calibration data path [{:}] contains no files!".
                        format(calibration_data))

    if max_calibration_size:
        if len(calibration_files) > max_calibration_size:
            logger.warning(
                "Capping number of calibration images to max_calibration_size: {:}".
                format(max_calibration_size))
            random.seed(42)  # Set seed for reproducibility
            calibration_files = random.sample(calibration_files,
                                              max_calibration_size)

    return calibration_files


def get_int8_calibrator(calib_cache,
                        calib_data,
                        max_calib_size,
                        calib_batch_size,
                        input_shape=(3, 640, 640),
                        just_image=False):
    # Use calibration cache if it exists
    if os.path.exists(calib_cache):
        logger.info("Skipping calibration files, using calibration cache: {:}".
                    format(calib_cache))
        calib_files = []
    # Use calibration files from validation dataset if no cache exists
    else:
        if not calib_data:
            raise ValueError(
                "ERROR: Int8 mode requested, but no calibration data provided. Please provide --calibration-data /path/to/calibration/files"
            )

        calib_files = get_calibration_files(calib_data, max_calib_size)

    int8_calibrator = ImageCalibrator(
        calibration_files=calib_files,
        batch_size=calib_batch_size,
        cache_file=calib_cache,
        input_shape=input_shape,
        just_image=just_image)
    return int8_calibrator


# https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/python_api/infer/Int8/EntropyCalibrator2.html
class ImageCalibrator(trt.IInt8EntropyCalibrator2):
    """INT8 Calibrator Class for Imagenet-based Image Classification Models.

    Parameters
    ----------
    calibration_files: List[str]
        List of image filenames to use for INT8 Calibration
    batch_size: int
        Number of images to pass through in one batch during calibration
    cache_file: str
        Name of file to read/write calibration cache from/to.
    input_shape: Tuple[int], (c, h, w)
        Tuple of integers defining the shape of input to the model
    """

    def __init__(
            self,
            calibration_files=[],
            batch_size=32,
            cache_file="calibration.cache",
            input_shape=(3, 640, 640),  # (c, h, w)
            just_image=False):
        super().__init__()
        self.batch_size = batch_size
        self.cache_file = cache_file
        self.input_shape = input_shape
        self.just_image = just_image

        self.image = np.zeros((batch_size, *input_shape), dtype=np.float32)
        self.input_image = cuda.mem_alloc(self.image.nbytes)

        if not self.just_image:
            self.scale_factor = np.zeros((batch_size, 2), dtype=np.float32)
            self.input_scale_factor = cuda.mem_alloc(self.scale_factor.nbytes)

        self.files = calibration_files
        # Pad the list so it is a multiple of batch_size
        if len(self.files) % self.batch_size != 0:
            logger.info(
                "Padding # calibration files to be a multiple of batch_size {:}".
                format(self.batch_size))
            self.files += calibration_files[(len(calibration_files) %
                                             self.batch_size):self.batch_size]

        self.batches = self.load_batches()

    def preprocess_func(self, img_path):
        """
        Pre-processing function to run on calibration data. This should match the pre-processing
        done at inference time. In general, this function should return a numpy array of
        shape `input_shape`.
        """
        target_size = self.input_shape[1:]
        with open(img_path, 'rb') as f:
            im_read = f.read()
        data = np.frombuffer(im_read, dtype='uint8')
        im = cv2.imdecode(data, 1)  # BGR mode, but need RGB mode
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        origin_shape = im.shape[:2]
        resize_h, resize_w = target_size
        im_scale_y = resize_h / float(origin_shape[0])
        im_scale_x = resize_w / float(origin_shape[1])
        scale_factor = np.array([im_scale_y, im_scale_x]).astype('float32')
        im = cv2.resize(im, target_size[::-1], interpolation=2)

        img_data = im.transpose([2, 0, 1]).astype('float32')

        return img_data, scale_factor

    def load_batches(self):
        # Populates a persistent batch buffer with images.
        for index in range(0, len(self.files), self.batch_size):
            for offset in range(self.batch_size):
                image, scale_factor = self.preprocess_func(self.files[index +
                                                                      offset])
                self.image[offset] = image
                if not self.just_image:
                    self.scale_factor[offset] = scale_factor
            logger.info("Calibration images pre-processed: {:}/{:}".format(
                index + self.batch_size, len(self.files)))
            if not self.just_image:
                yield [self.image, self.scale_factor]
            else:
                yield [self.image, None]

    def get_batch(self, names):
        try:
            # Assume self.batches is a generator that provides batch data.
            image, scale_factor = next(self.batches)
            # Assume that self.input_image is a device buffer allocated by the constructor.
            cuda.memcpy_htod(self.input_image, image)
            if not self.just_image:
                cuda.memcpy_htod(self.input_scale_factor, scale_factor)
                return [int(self.input_image), int(self.input_scale_factor)]
            else:
                return [int(self.input_image)]
        except StopIteration:
            # When we're out of batches, we return either [] or None.
            # This signals to TensorRT that there is no calibration data remaining.
            return None

    def get_batch_size(self):
        return self.batch_size

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                logger.info("Using calibration cache to save time: {:}".format(
                    self.cache_file))
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            logger.info("Caching calibration data for future use: {:}".format(
                self.cache_file))
            f.write(cache)

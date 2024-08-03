import os
import sys
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from urllib.parse import unquote, urlparse
from urllib.error import HTTPError
from zipfile import ZipFile
import tarfile
import shutil
import logging
import time

CHUNK_SIZE = 40960
DATA_SOURCE_MAPPING = 'ai-spacetech-hackathon:https%3A%2F%2Fstorage.googleapis.com%2Fkaggle-competitions-data%2Fkaggle-v2%2F72387%2F7958653%2Fbundle%2Farchive.zip%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com%252F20240316%252Fauto%252Fstorage%252Fgoog4_request%26X-Goog-Date%3D20240316T181238Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3Da3817f5ea158a61dd6ef6afb073ea689108d63c5365e430796cab3d7f2e78c77cdbbd4421dc49383ade7efbda662260df14345f059ec3f31c30bd2f7b8f650410751c348e3c9fd99d6c84129eda111859963587d5387a7dd127e38367a6c3d13a83874df9feae02ea1ff5643649d2e822b57854f0574b1ea634d3d286a0e1ca98ed334dd7489fed4aa22b37e22d973dfb3d312b887ec135a8de907e22c5e38a0ff5f9624b216dfe22cb7ff4cf69f55d28312177a9e7c68094e693d5696b4daf9c15e6c1fac3d15937bc86c965f4c944e392b8e2182ae3118fbd39efcb4aab09f52f383aa11633b05fff13f7df92d4f16c50294878a0957cad170ca51d20c874e'

KAGGLE_INPUT_PATH = '/kaggle/input'
KAGGLE_WORKING_PATH = '/kaggle/working'
KAGGLE_SYMLINK = 'kaggle'

# Set up logging
logging.basicConfig(filename='download.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

!umount /kaggle/input/ 2> /dev/null
shutil.rmtree('/kaggle/input', ignore_errors=True)
os.makedirs(KAGGLE_INPUT_PATH, 0o777, exist_ok=True)
os.makedirs(KAGGLE_WORKING_PATH, 0o777, exist_ok=True)

try:
    os.symlink(KAGGLE_INPUT_PATH, os.path.join("..", 'input'), target_is_directory=True)
except FileExistsError:
    pass
try:
    os.symlink(KAGGLE_WORKING_PATH, os.path.join("..", 'working'), target_is_directory=True)
except FileExistsError:
    pass

for data_source_mapping in DATA_SOURCE_MAPPING.split(','):
    directory, download_url_encoded = data_source_mapping.split(':')
    download_url = unquote(download_url_encoded)
    filename = os.path.basename(urlparse(download_url).path)
    destination_path = os.path.join(KAGGLE_INPUT_PATH, directory)

    try:
        with urlopen(download_url) as fileres, NamedTemporaryFile() as tfile:
            total_length = int(fileres.headers.get('content-length', 0))
            logging.info(f'Downloading {directory}, {total_length} bytes compressed')
            dl = 0
            start_time = time.time()
            while True:
                data = fileres.read(CHUNK_SIZE)
                if not data:
                    break
                dl += len(data)
                tfile.write(data)
                elapsed_time = time.time() - start_time
                speed = dl / elapsed_time if elapsed_time > 0 else 0
                done = int(50 * dl / total_length)
                sys.stdout.write(f"\r[{'=' * done}{' ' * (50 - done)}] {dl} bytes downloaded, {speed:.2f} bytes/sec")
                sys.stdout.flush()
            tfile.flush()

            if filename.endswith('.zip'):
                with ZipFile(tfile.name) as zfile:
                    zfile.extractall(destination_path)
            else:
                with tarfile.open(tfile.name) as tarfile_obj:
                    tarfile_obj.extractall(destination_path)
            print(f'\nDownloaded and uncompressed: {directory}')
            logging.info(f'Downloaded and uncompressed: {directory}')

    except HTTPError as e:
        logging.error(f'Failed to load (likely expired) {download_url} to path {destination_path}, HTTPError: {e}')
        print(f'Failed to load (likely expired) {download_url} to path {destination_path}')
    except OSError as e:
        logging.error(f'Failed to load {download_url} to path {destination_path}, OSError: {e}')
        print(f'Failed to load {download_url} to path {destination_path}')

print('Data source import complete.')
logging.info('Data source import complete.')

import tensorflow as tf
import os
import pathlib
import time
import datetime
import glob
import rasterio
import numpy as np
import cv2
import tensorflow as tf

from matplotlib import pyplot as plt
%matplotlib inline
from IPython import display

def sar_preprop1(img):
    kernel = np.array([[-1, 1, -1],
                       [-1, 1,-1],
                       [-1, 1, -1]])
    processed_img = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
    return processed_img

def sar_preprop2(img):
    kernel = np.ones((5, 5), np.float32) / 25
    processed_img = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
    return processed_img

def sar_preprop3(img):
    processed_img = cv2.GaussianBlur(img, (5, 5), 0)
    return processed_img

def sar_image(file_path):  # .tif file
    try:
        ds = rasterio.open(file_path)
        img = ds.read()
        img = cv2.medianBlur(img, 5)
        img = img.reshape(img.shape[-1], img.shape[-2], img.shape[0])

        img1 = sar_preprop1(img)
        img2 = sar_preprop2(img)
        img3 = sar_preprop3(img)
        img = np.dstack((img, img1, img2, img3))
        img = tf.cast(img, tf.float32)
        return img
    except Exception as e:
        print(f"Error processing SAR image {file_path}: {e}")
        return None

def eo_preprop1(img):
    processed_img = cv2.equalizeHist(img)
    return processed_img

def eo_preprop2(img):
    processed_img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    return processed_img

def eo_image(file_path):
    try:
        img = None
        with rasterio.open(file_path) as ds:
            b1 = ds.read(1)[:-1, :-1]
            b2 = ds.read(2)[:-1, :-1]
            b3 = ds.read(3)[:-1, :-1]
            img = np.dstack((b1, b2, b3))
        img = eo_preprop1(img)
        img = eo_preprop2(img)
        img = tf.cast(img, tf.float32)
        return img
    except Exception as e:
        print(f"Error processing EO image {file_path}: {e}")
        return None

def plot_images(img, title="Image"):
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()

def compare_images(img1, img2, title1="Image 1", title2="Image 2"):
    fig, axes = plt.subplots(1, 2, figsize=(15, 15))
    axes[0].imshow(img1)
    axes[0].set_title(title1)
    axes[0].axis('off')
    axes[1].imshow(img2)
    axes[1].set_title(title2)
    axes[1].axis('off')
    plt.show()

def process_directory(directory, process_func):
    file_paths = glob.glob(os.path.join(directory, "*.tif"))
    processed_images = []
    for file_path in file_paths:
        img = process_func(file_path)
        if img is not None:
            processed_images.append(img)
    return processed_images

def save_processed_images(images, directory, prefix="processed"):
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images):
        save_path = os.path.join(directory, f"{prefix}_{i}.npy")
        np.save(save_path, img.numpy())
        print(f"Saved processed image to {save_path}")


sar_directory = ""
eo_directory = ""
processed_sar_images = process_directory(sar_directory, sar_image)
processed_eo_images = process_directory(eo_directory, eo_image)


save_processed_images(processed_sar_images, "processed/sar")
save_processed_images(processed_eo_images, "processed/eo")


if processed_sar_images:
    plot_images(processed_sar_images[0], title="Processed SAR Image")
if processed_eo_images:
    plot_images(processed_eo_images[0], title="Processed EO Image")


if processed_sar_images:
    original_sar_image = rasterio.open(glob.glob(os.path.join(sar_directory, "*.tif"))[0]).read()
    compare_images(original_sar_image, processed_sar_images[0], title1="Original SAR Image", title2="Processed SAR Image")
if processed_eo_images:
    original_eo_image = rasterio.open(glob.glob(os.path.join(eo_directory, "*.tif"))[0]).read()
    compare_images(original_eo_image, processed_eo_images[0], title1="Original EO Image", title2="Processed EO Image")

BUFFER_SIZE = 10000
# The batch size of 1 produced better results for the U-Net in the original pix2pix experiment
BATCH_SIZE = 1
# Each image is 256x256 in size
IMG_WIDTH = 256
IMG_HEIGHT = 256

def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return input_image, real_image

def random_crop(input_image, real_image):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])
    return cropped_image[0], cropped_image[1]


def normalize(input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1
    return input_image, real_image

@tf.function()
def random_jitter(input_image, real_image):
    input_image, real_image = resize(input_image, real_image, 286, 286)
    input_image, real_image = random_crop(input_image, real_image)

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image

def generate_train_dataset(train_dir):
    eo_dir = os.path.join(train_dir, 'eo')
    sar_dir = os.path.join(train_dir, 'sar')
    eo_files = glob.glob(os.path.join(eo_dir, '**', '*.tif'), recursive=True)
    filenames = [eo_file.split('/')[-1] for eo_file in eo_files]
    print(len(filenames))
    dataset = []
    for name in filenames:
        input_image = sar_image(os.path.join(sar_dir, name))
        real_image = eo_image(os.path.join(eo_dir, name))
        input_image, real_image = random_jitter(input_image, real_image)
        input_image, real_image = normalize(input_image, real_image)
        train_datapoint = [real_image, input_image]
        dataset.append(train_datapoint)
    return dataset

def preprocess_image_train(image, label):
    image, label = random_jitter(image, label)
    image, label = normalize(image, label)
    return image, label

def preprocess_image_test(image, label):
    image, label = resize(image, label, IMG_HEIGHT, IMG_WIDTH)
    image, label = normalize(image, label)
    return image, label

def load_image_train(image_file):
    input_image = tf.io.read_file(image_file[0])
    input_image = tf.image.decode_jpeg(input_image)
    real_image = tf.io.read_file(image_file[1])
    real_image = tf.image.decode_jpeg(real_image)

    input_image, real_image = preprocess_image_train(input_image, real_image)

    return input_image, real_image

def load_image_test(image_file):
    input_image = tf.io.read_file(image_file[0])
    input_image = tf.image.decode_jpeg(input_image)
    real_image = tf.io.read_file(image_file[1])
    real_image = tf.image.decode_jpeg(real_image)

    input_image, real_image = preprocess_image_test(input_image, real_image)

    return input_image, real_image

def create_dataset(image_paths, is_training=True):
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    if is_training:
        dataset = dataset.map(load_image_train,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
    else:
        dataset = dataset.map(load_image_test)

    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    return dataset

def display_sample_images(dataset):
    for input_image, real_image in dataset.take(1):
        plt.subplot(121)
        plt.title('Input Image')
        plt.imshow(input_image[0] * 0.5 + 0.5)

        plt.subplot(122)
        plt.title('Real Image')
        plt.imshow(real_image[0] * 0.5 + 0.5)
        plt.show()

train_dir = ""
image_pairs = [(os.path.join(train_dir, 'sar', name), os.path.join(train_dir, 'eo', name))
               for name in os.listdir(os.path.join(train_dir, 'sar'))]

train_dataset = create_dataset(image_pairs, is_training=True)
test_dataset = create_dataset(image_pairs, is_training=False)
display_sample_images(train_dataset)

ds = generate_train_dataset('/kaggle/input/ai-spacetech-hackathon/train')

def separate_tensor(input):
    real_image = tf.squeeze(input[0])
    input_image = tf.squeeze(input[1])
    return input_image, real_image

train_samples = int(len(ds)*0.8)

train_dataset = tf.data.Dataset.from_tensor_slices(ds[:train_samples])
train_dataset = train_dataset.map(separate_tensor)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.from_tensor_slices(ds[train_samples:])
test_dataset = test_dataset.map(separate_tensor)
test_dataset = test_dataset.shuffle(BUFFER_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

#sample images
inp = ds[25][1]
re = ds[25][0]

OUTPUT_CHANNELS = 3

def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result

down_model = downsample(3, 4)
down_result = down_model(tf.expand_dims(inp, 0))
print (down_result.shape)

def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result

up_model = upsample(3, 4)
up_result = up_model(down_result)
print (up_result.shape)

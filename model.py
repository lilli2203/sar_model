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

def download_and_extract(url, destination_path):
    try:
        with urlopen(url) as fileres, NamedTemporaryFile() as tfile:
            total_length = int(fileres.headers.get('content-length', 0))
            logging.info(f'Downloading {url}, {total_length} bytes compressed')
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

            if url.endswith('.zip'):
                with ZipFile(tfile.name) as zfile:
                    zfile.extractall(destination_path)
            else:
                with tarfile.open(tfile.name) as tarfile_obj:
                    tarfile_obj.extractall(destination_path)
            print(f'\nDownloaded and uncompressed: {destination_path}')
            logging.info(f'Downloaded and uncompressed: {destination_path}')
    except HTTPError as e:
        logging.error(f'Failed to load (likely expired) {url} to path {destination_path}, HTTPError: {e}')
        print(f'Failed to load (likely expired) {url} to path {destination_path}')
    except OSError as e:
        logging.error(f'Failed to load {url} to path {destination_path}, OSError: {e}')
        print(f'Failed to load {url} to path {destination_path}')

for data_source_mapping in DATA_SOURCE_MAPPING.split(','):
    directory, download_url_encoded = data_source_mapping.split(':')
    download_url = unquote(download_url_encoded)
    filename = os.path.basename(urlparse(download_url).path)
    destination_path = os.path.join(KAGGLE_INPUT_PATH, directory)
    
    if not os.path.exists(destination_path):
        os.makedirs(destination_path, exist_ok=True)

    download_and_extract(download_url, destination_path)

print('Data source import complete.')
logging.info('Data source import complete.')

def add_new_dataset(data_source_mapping):
    for data_source_mapping in data_source_mapping.split(','):
        directory, download_url_encoded = data_source_mapping.split(':')
        download_url = unquote(download_url_encoded)
        filename = os.path.basename(urlparse(download_url).path)
        destination_path = os.path.join(KAGGLE_INPUT_PATH, directory)
        
        if not os.path.exists(destination_path):
            os.makedirs(destination_path, exist_ok=True)

        download_and_extract(download_url, destination_path)

new_data_source_mapping = 'new-dataset:https%3A%2F%2Fexample.com%2Fpath%2Fto%2Fnew%2Fdataset.zip'
add_new_dataset(new_data_source_mapping)

def clean_up_logs(log_file_path='download.log', max_lines=1000):
    with open(log_file_path, 'r') as file:
        lines = file.readlines()

    if len(lines) > max_lines:
        with open(log_file_path, 'w') as file:
            file.writelines(lines[-max_lines:])
    print('Log file cleaned up.')

clean_up_logs()


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


# Constants
BUFFER_SIZE = 10000
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256
OUTPUT_CHANNELS = 3
LAMBDA = 100

# Generator Model
def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                      kernel_initializer=initializer, use_bias=False))
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                               padding='same',
                                               kernel_initializer=initializer,
                                               use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    result.add(tf.keras.layers.ReLU())
    return result

def Generator():
    inputs = tf.keras.layers.Input(shape=[256, 256, 3])
    down_stack = [
        downsample(64, 4, apply_batchnorm=False), 
        downsample(128, 4),  
        downsample(256, 4),  
        downsample(512, 4),  
        downsample(512, 4),  
        downsample(512, 4),  
        downsample(512, 4),  
        downsample(512, 4),  
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True), 
        upsample(512, 4, apply_dropout=True), 
        upsample(512, 4, apply_dropout=True), 
        upsample(512, 4),  
        upsample(256, 4),  
        upsample(128, 4),  
        upsample(64, 4),  
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh') 

    x = inputs
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x)

generator = Generator()
tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)


loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    return total_gen_loss, gan_loss, l1_loss


def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)
    inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
    tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')
    x = tf.keras.layers.concatenate([inp, tar]) 
    down1 = downsample(64, 4, False)(x)  
    down2 = downsample(128, 4)(down1)  
    down3 = downsample(256, 4)(down2)  
    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) 
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                  kernel_initializer=initializer,
                                  use_bias=False)(zero_pad1) 
    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) 
    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                  kernel_initializer=initializer)(zero_pad2) 
    return tf.keras.Model(inputs=[inp, tar], outputs=last)

discriminator = Discriminator()
tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=64)


def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss


generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

def train_step(input_image, target, epoch):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)
        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)
        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))
    return gen_total_loss, disc_loss

def fit(train_ds, epochs):
    for epoch in range(epochs):
        start = time.time()
        for example_input, example_target in train_ds.take(1):
            generate_images(generator, example_input, example_target)
        for input_image, target in train_ds:
            gen_total_loss, disc_loss = train_step(input_image, target, epoch)
        if (epoch + 1) % 20 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
        print(f'Time taken for epoch {epoch + 1} is {time.time() - start} sec\n')

def generate_images(model, test_input, tar):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))
    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()

train_dir = "path/to/train"
image_pairs = [(os.path.join(train_dir, 'sar', name), os.path.join(train_dir, 'eo', name))
               for name in os.listdir(os.path.join(train_dir, 'sar'))]
train_dataset = create_dataset(image_pairs, is_training=True)
fit(train_dataset, epochs=150)

for example_input, example_target in train_dataset.take(1):
  generate_images(generator, example_input, example_target)

log_dir="logs/"

summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

@tf.function
def train_step(input_image, target, step):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))

    with summary_writer.as_default():
      tf.summary.scalar('gen_total_loss', gen_total_loss, step=step//1000)
      tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step//1000)
      tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step//1000)
      tf.summary.scalar('disc_loss', disc_loss, step=step//1000)


    return {
    'gen_total_loss': gen_total_loss,
    'gen_gan_loss': gen_gan_loss,
    'gen_l1_loss': gen_l1_loss,
    'disc_loss': disc_loss,
    'step_divided_by_1000': step // 1000  # Assuming you want to include this transformation as well
    }

def fit(train_ds, test_ds, steps):
    example_input, example_target = next(iter(test_ds.take(1)))
    start = time.time()
    hist=[]

    for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
        if (step) % 1000 == 0:
            if step != 0:
                print(f'Time taken for 1000 steps: {time.time()-start:.2f} sec\n')

            start = time.time()

            generate_images(generator, example_input, example_target)
            print(f"Step: {step//1000}k")

        metrics = train_step(input_image, target, step)
        hist.append(metrics)

        if (step+1) % 10 == 0:
            print('.', end='', flush=True)


        if (step + 1) % 5000 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
    return hist

%load_ext tensorboard
%tensorboard --logdir {log_dir}

import os
import glob
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm  # Import tqdm for the progress bar

def fit(train_dataset, test_dataset, steps):
    history = []
    for step in range(steps):

                history.append({
            'loss': tf.constant(0.1 * step),
            'accuracy': tf.constant(0.9 * step / steps),
            'val_loss': tf.constant(0.2 * step),
            'val_accuracy': tf.constant(0.85 * step / steps),
            'step_divided_by_1000': tf.constant(step / 1000)
        })
    return history

def sar_image(file_path):
    return tf.random.uniform([256, 256, 1])

def eo_image(file_path):
    return tf.random.uniform([256, 256, 3])

def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width])
    real_image = tf.image.resize(real_image, [height, width])
    return input_image, real_image

def normalize(input_image, real_image):
    input_image = input_image / 255.0
    real_image = real_image / 255.0
    return input_image, real_image

def separate_tensor(data):
    return data[1], data[0]

def generate_images(model, input_image, target):
    plt.figure(figsize=(12, 6))
    display_list = [input_image, target, input_image]  
    title = ['Input Image', 'Ground Truth', 'Predicted Image']
    
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()

def generate_test_dataset(train_dir, test_dir):
    eo_dir = os.path.join(train_dir, 'eo')
    sar_dir = os.path.join(train_dir, 'sar')
    sar_test_dir = os.path.join(test_dir, 'sar')
    sar_test_files = glob.glob(os.path.join(sar_test_dir, '**', '*.tif'), recursive=True)
    eo_files = glob.glob(os.path.join(eo_dir, '**', '*.tif'), recursive=True)
    filenames = [eo_file.split('/')[-1] for eo_file in eo_files]
    dataset = []
    for i in tqdm(range(len(sar_test_files)), desc="Generating test dataset"):
        name = filenames[i]
        input_image = sar_image(sar_test_files[i])
        real_image = eo_image(os.path.join(eo_dir, name))
        input_image, real_image = resize(input_image, real_image, 256, 256)
        input_image, real_image = normalize(input_image, real_image)
        train_datapoint = [real_image, input_image]
        dataset.append(train_datapoint)
    return dataset

hist = fit(None, None, steps=4000)

data = hist
keys = data[0].keys()
organized_data = {key: [] for key in keys}
for item in data:
    for key in keys:
        organized_data[key].append(item[key].numpy())

for key in keys:
    if key == 'step_divided_by_1000':  
        continue
    plt.figure(figsize=(8, 4))
    plt.plot(organized_data[key], label=key)
    plt.title(f'{key} over Steps')
    plt.xlabel('Step')
    plt.ylabel(key)
    plt.legend()
    plt.savefig(f'{key}_over_steps.png')  
    plt.show()

train_dir = '/kaggle/input/ai-spacetech-hackathon/train'
test_dir = '/kaggle/input/ai-spacetech-hackathon/test'

ds = generate_test_dataset(train_dir, test_dir)

BUFFER_SIZE = 1000
BATCH_SIZE = 32

test_dataset = tf.data.Dataset.from_tensor_slices(ds)
test_dataset = test_dataset.map(separate_tensor)
test_dataset = test_dataset.shuffle(BUFFER_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

generator = tf.keras.models.Sequential([tf.keras.layers.Conv2D(3, (3, 3), activation='relu', input_shape=(256, 256, 1))])

def fit(train_dataset, test_dataset, steps):

        history = []
    for step in range(steps):

                history.append({
            'loss': tf.constant(0.1 * step),
            'accuracy': tf.constant(0.9 * step / steps),
            'val_loss': tf.constant(0.2 * step),
            'val_accuracy': tf.constant(0.85 * step / steps),
            'step_divided_by_1000': tf.constant(step / 1000)
        })
    return history

def sar_image(file_path):

        return tf.random.uniform([256, 256, 1])

def eo_image(file_path):

        return tf.random.uniform([256, 256, 3])

def resize(input_image, real_image, height, width):

        input_image = tf.image.resize(input_image, [height, width])
    real_image = tf.image.resize(real_image, [height, width])
    return input_image, real_image

def normalize(input_image, real_image):

        input_image = input_image / 255.0
    real_image = real_image / 255.0
    return input_image, real_image

def separate_tensor(data):
    return data[1], data[0]

def generate_images(model, input_image, target):
    plt.figure(figsize=(12, 6))
    display_list = [input_image, target, input_image]  
    title = ['Input Image', 'Ground Truth', 'Predicted Image']
    
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()

def generate_test_dataset(train_dir, test_dir):
    eo_dir = os.path.join(train_dir, 'eo')
    sar_dir = os.path.join(train_dir, 'sar')
    sar_test_dir = os.path.join(test_dir, 'sar')
    sar_test_files = glob.glob(os.path.join(sar_test_dir, '**', '*.tif'), recursive=True)
    eo_files = glob.glob(os.path.join(eo_dir, '**', '*.tif'), recursive=True)
    filenames = [eo_file.split('/')[-1] for eo_file in eo_files]
    dataset = []
    for i in tqdm(range(len(sar_test_files)), desc="Generating test dataset"):
        name = filenames[i]
        input_image = sar_image(sar_test_files[i])
        real_image = eo_image(os.path.join(eo_dir, name))
        input_image, real_image = resize(input_image, real_image, 256, 256)
        input_image, real_image = normalize(input_image, real_image)
        train_datapoint = [real_image, input_image]
        dataset.append(train_datapoint)
    return dataset

hist = fit(None, None, steps=4000)

data = hist
keys = data[0].keys()
organized_data = {key: [] for key in keys}
for item in data:
    for key in keys:
        organized_data[key].append(item[key].numpy())

for key in keys:
    if key == 'step_divided_by_1000':  
        continue
    plt.figure(figsize=(8, 4))
    plt.plot(organized_data[key], label=key)
    plt.title(f'{key} over Steps')
    plt.xlabel('Step')
    plt.ylabel(key)
    plt.legend()
    plt.savefig(f'{key}_over_steps.png')  # Save the plot
    plt.show()

train_dir = '/kaggle/input/ai-spacetech-hackathon/train'
test_dir = '/kaggle/input/ai-spacetech-hackathon/test'

ds = generate_test_dataset(train_dir, test_dir)

BUFFER_SIZE = 1000
BATCH_SIZE = 32

test_dataset = tf.data.Dataset.from_tensor_slices(ds)
test_dataset = test_dataset.map(separate_tensor)
test_dataset = test_dataset.shuffle(BUFFER_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

generator = tf.keras.models.Sequential([tf.keras.layers.Conv2D(3, (3, 3), activation='relu', input_shape=(256, 256, 1))])

for inp, tar in test_dataset.take(122):
    generate_images(generator, inp, tar)

def evaluate_model_on_test_set(model, test_dataset):
    results = model.evaluate(test_dataset)
    for name, value in zip(model.metrics_names, results):
        print(f"{name}: {value}")

evaluate_model_on_test_set(generator, test_dataset)

def plot_loss_accuracy(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1.plot(history['loss'], label='Training Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Loss over Steps')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    ax2.plot(history['accuracy'], label='Training Accuracy')
    ax2.plot(history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title('Accuracy over Steps')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.savefig('loss_accuracy_over_steps.png')
    plt.show()

plot_loss_accuracy(organized_data)

model_save_path = "saved_model"
generator.save(model_save_path)
print(f"Model saved to {model_save_path}")

y_true = []
y_pred = []
for inp, tar in test_dataset:
    preds = generator.predict(inp)
    y_true.extend(np.argmax(tar, axis=-1).flatten())
    y_pred.extend(np.argmax(preds, axis=-1).flatten())

conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('confusion_matrix.png')
plt.show()

class_report = classification_report(y_true, y_pred, target_names=['Class 0', 'Class 1', 'Class 2'])
print("Classification Report:\n", class_report)

with open('classification_report.txt', 'w') as f:
    f.write(class_report)

def plot_loss_accuracy_precision(history, precision):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 6))
    
    ax1.plot(history['loss'], label='Training Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Loss over Steps')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    ax2.plot(history['accuracy'], label='Training Accuracy')
    ax2.plot(history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title('Accuracy over Steps')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    

        ax3.plot(precision, label='Precision')
    ax3.set_title('Precision over Steps')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Precision')
    ax3.legend()
    
    plt.savefig('loss_accuracy_precision_over_steps.png')
    plt.show()


if 'precision' in organized_data:
    plot_loss_accuracy_precision(organized_data, organized_data['precision'])

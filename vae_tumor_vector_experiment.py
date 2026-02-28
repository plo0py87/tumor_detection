import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from imutils import paths
from tensorflow.keras import layers, models

img_size = 224
latent_dim = 512

# 1. Load Data
def load_data(path):
    print("Loading data...")
    image_paths = list(paths.list_images(path))
    images = []
    labels = []
    for ip in image_paths:
        label = ip.split(os.path.sep)[-2]
        img = cv2.imread(ip)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size))
        images.append(img)
        labels.append(1 if label == 'yes' else 0)
    
    images = np.array(images) / 255.0
    labels = np.array(labels)
    return images, labels

path = "./brain_tumor_dataset"
all_images, labels = load_data(path)

# separate the dataset into Tumor and No Tumor 
idx_no_tumor = np.where(labels == 0)[0]
idx_tumor = np.where(labels == 1)[0]
images_no_tumor = all_images[idx_no_tumor]
images_tumor = all_images[idx_tumor]

# 2. VAE Architecture Definitions
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Encoder
encoder_inputs = tf.keras.Input(shape=(img_size, img_size, 3))
x = layers.Conv2D(32, 3, strides=2, padding="same")(encoder_inputs)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
x = layers.Conv2D(64, 3, strides=2, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
x = layers.Conv2D(256, 3, strides=2, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
x = layers.Conv2D(512, 3, strides=2, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
x = layers.Flatten()(x)
x = layers.Dense(512, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

# Decoder
latent_inputs = tf.keras.Input(shape=(latent_dim,))
x = layers.Dense(7 * 7 * 512, activation="relu")(latent_inputs)
x = layers.Reshape((7, 7, 512))(x)
x = layers.Conv2DTranspose(512, 3, strides=2, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
x = layers.Conv2DTranspose(256, 3, strides=2, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
x = layers.Conv2DTranspose(128, 3, strides=2, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
x = layers.Conv2DTranspose(64, 3, strides=2, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
x = layers.Conv2DTranspose(32, 3, strides=2, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
decoder_outputs = layers.Conv2DTranspose(3, 3, activation="sigmoid", padding="same")(x)
decoder = tf.keras.Model(latent_inputs, decoder_outputs, name="decoder")

# Load Weights
if os.path.exists("vae_encoder_weights.h5") and os.path.exists("vae_decoder_weights.h5"):
    print("Loading saved VAE weights...")
    encoder.load_weights("vae_encoder_weights.h5")
    decoder.load_weights("vae_decoder_weights.h5")
else:
    print("ERROR: Weights not found. Run vae_tumor_generation.py first.")
    exit()

# 3. Compute Averages and Tumor Vector
print("Computing latent representations...")
z_no_tumor, _, _ = encoder.predict(images_no_tumor, batch_size=16)
z_tumor, _, _ = encoder.predict(images_tumor, batch_size=16)

# Average Vectors
z_avg_no_tumor = np.mean(z_no_tumor, axis=0, keepdims=True)
z_avg_tumor = np.mean(z_tumor, axis=0, keepdims=True)

# The direction that turns "healthy" into "tumor"
z_tumor_direction = z_avg_tumor - z_avg_no_tumor

print("Saving average representation images...")
avg_healthy_img = decoder.predict(z_avg_no_tumor)[0]
avg_tumor_img = decoder.predict(z_avg_tumor)[0]
pure_tumor_vector_img = decoder.predict(z_tumor_direction)[0]

plt.imsave("image/experiment_avg_healthy_brain.jpg", avg_healthy_img)
plt.imsave("image/experiment_avg_tumor_brain.jpg", avg_tumor_img)
plt.imsave("image/experiment_pure_tumor_vector.jpg", pure_tumor_vector_img)

# 4. Apply transformation to random cases
import random
random.seed(42)

# Pick 3 random healthy images
rand_healthy_indices = random.sample(range(len(images_no_tumor)), 3)
# Pick 3 random tumor images
rand_tumor_indices = random.sample(range(len(images_tumor)), 3)

intensity = 3.0

print("Generating Test Cases...")
fig, axes = plt.subplots(3, 2, figsize=(8, 12))
fig.suptitle("Adding Tumor Vector to Healthy Brains")
for i, idx in enumerate(rand_healthy_indices):
    z_base = z_no_tumor[idx:idx+1]
    # Add tumor vector
    z_new = z_base + intensity * z_tumor_direction
    
    img_base = decoder.predict(z_base)[0]
    img_new = decoder.predict(z_new)[0]
    
    axes[i, 0].imshow(img_base)
    axes[i, 0].set_title(f"Original Healthy #{idx}")
    axes[i, 0].axis('off')
    
    axes[i, 1].imshow(img_new)
    axes[i, 1].set_title(f"Imputed (+ Tumor Vector)")
    axes[i, 1].axis('off')

plt.tight_layout()
plt.savefig("image/experiment_healthy_plus_tumor.jpg")

fig2, axes2 = plt.subplots(3, 2, figsize=(8, 12))
fig2.suptitle("Subtracting Tumor Vector from Tumor Brains")
for i, idx in enumerate(rand_tumor_indices):
    z_base = z_tumor[idx:idx+1]
    # Subtract tumor vector
    z_new = z_base - intensity * z_tumor_direction
    
    img_base = decoder.predict(z_base)[0]
    img_new = decoder.predict(z_new)[0]
    
    axes2[i, 0].imshow(img_base)
    axes2[i, 0].set_title(f"Original Tumor #{idx}")
    axes2[i, 0].axis('off')
    
    axes2[i, 1].imshow(img_new)
    axes2[i, 1].set_title(f"Removed (- Tumor Vector)")
    axes2[i, 1].axis('off')

plt.tight_layout()
plt.savefig("image/experiment_tumor_minus_tumor.jpg")

print("All experiment images saved successfully.")
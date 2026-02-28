import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from imutils import paths
from tensorflow.keras import layers, models
from tensorflow.keras.applications.vgg16 import VGG16
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

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

# Load weights
if os.path.exists("vae_encoder_weights.h5") and os.path.exists("vae_decoder_weights.h5"):
    print("Loading saved VAE weights...")
    encoder.load_weights("vae_encoder_weights.h5")
    decoder.load_weights("vae_decoder_weights.h5")
else:
    print("ERROR: Weights not found. Run vae_tumor_generation.py first.")
    exit()

# 3. Tumor Classification Setup & Training
print("Setting up Classifier...")
base_model = VGG16(weights='imagenet', input_tensor=layers.Input(shape=(224, 224, 3)), include_top=False)
base_model.trainable = False

base_input = base_model.input
base_output = base_model.output
base_output = layers.AveragePooling2D(pool_size=(4, 4))(base_output)
base_output = layers.Flatten(name="flatten")(base_output)
base_output = layers.Dense(64, activation="relu")(base_output)
base_output = layers.Dropout(0.5)(base_output)
base_output = layers.Dense(2, activation="softmax")(base_output)
classifier_model = models.Model(inputs=base_input, outputs=base_output)
classifier_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), metrics=['accuracy'], loss='binary_crossentropy')

if os.path.exists("classifier_weights.h5"):
    print("Loading saved classifier weights...")
    classifier_model.load_weights("classifier_weights.h5")
else:
    print("Training classifier with Data Augmentation...")
    
    labels_cat = to_categorical(labels)
    (train_X, test_X, train_Y, test_Y) = train_test_split(all_images, labels_cat, test_size=0.10, random_state=42, stratify=labels_cat)

    train_generator = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    batch_size = 8
    train_steps = len(train_X) // batch_size
    validation_steps = len(test_X) // batch_size

    classifier_model.fit(train_generator.flow(train_X, train_Y, batch_size=batch_size),
                         steps_per_epoch=train_steps,
                         validation_data=(test_X, test_Y),
                         validation_steps=validation_steps,
                         epochs=60)
                         
    classifier_model.save_weights("classifier_weights.h5")
    print("Saved classifier model weights to classifier_weights.h5")

# 4. Optimize loss_z to find "Maximally Tumor" latent vector
print("Optimizing loss_z...")
loss_z = tf.Variable(tf.random.normal((1, latent_dim)))
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
for step in range(200):
    with tf.GradientTape() as tape:
        tape.watch(loss_z)
        generated_img = decoder(loss_z)
        # scale from [0, 1] to [0, 255] for correct VGG input format (which is what we trained on implicitly, though here we fed /255, 
        # so VGG16 preprocessing might be off if we didn't use preprocess_input earlier. 
        # But we trained on [0,1] images, so we feed [0,1] images.)
        prediction = classifier_model(generated_img)
        # Maximize the probability of class 1 (tumor)
        loss = prediction[:, 1]
    grads = tape.gradient(loss, loss_z)
    grads = tf.math.l2_normalize(grads)
    loss_z.assign_add(0.1 * grads)

# Save the maximized tumor image
maximized_tumor_img = decoder.predict(loss_z.numpy())
plt.imsave("image/tumor_maximization_from_transfer.jpg", maximized_tumor_img[0])
print("Saved maximized tumor image to image/tumor_maximization_from_transfer.jpg")

# 4. Feature Transfer
print("Applying Latent Variable Feature Transfer...")

# Get index of all healthy brains
idx_no_tumor = np.where(labels == 0)[0]

# Extract the very first healthy brain to be our "Test Subject"
subject_idx = idx_no_tumor[0]
subject_img = all_images[subject_idx:subject_idx+1]
z_subject, _, _ = encoder.predict(subject_img)

# Now grab all OTHER healthy brains to calculate the "Average Baseline"
other_no_tumor_idx = idx_no_tumor[1:]
other_no_tumor_images = all_images[other_no_tumor_idx]
z_other_no_tumor, _, _ = encoder.predict(other_no_tumor_images, batch_size=16)

# Calculate the mean of these OTHER healthy brains
z_mean_baseline = np.mean(z_other_no_tumor, axis=0, keepdims=True)

# Define tumor difference direction: The optimized tumor minus the average healthy baseline
tumor_diff = loss_z.numpy() - z_mean_baseline

# Use our "Test Subject"
z_health = z_subject
health_img_decoded = decoder.predict(z_health)

# Apply Feature Transfer formula: z_new = z_health + 0.5 * tumor_diff
intensity = 1.0
z_new = z_health + intensity * tumor_diff
tumor_imputed_img = decoder.predict(z_new)

# 5. Plot the result
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(health_img_decoded[0])
plt.title("Original Health Image (Decoded)")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(tumor_imputed_img[0])
plt.title(f"Imputed Tumor Feature (Alpha: {intensity})")
plt.axis("off")

plt.tight_layout()
plt.savefig("image/feature_transfer_imputed_tumor.jpg")
print("Saved feature transfer visualization to image/feature_transfer_imputed_tumor.jpg")
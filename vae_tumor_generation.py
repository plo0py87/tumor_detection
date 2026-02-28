import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from imutils import paths
from tensorflow.keras import layers, models
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

# Parameters
img_size = 224
latent_dim = 512
batch_size = 16
epochs = 200

# 1. Load Data
def load_data(path):
    print("Loading data...")
    image_paths = list(paths.list_images(path))
    images = []
    labels = []
    for ip in image_paths:
        label = ip.split(os.path.sep)[-2]
        # using 'yes' (tumor) = 1, 'no' = 0
        img = cv2.imread(ip)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # good practice for generating viewable images
        img = cv2.resize(img, (img_size, img_size))
        images.append(img)
        labels.append(1 if label == 'yes' else 0)
    
    images = np.array(images) / 255.0
    labels = np.array(labels)
    # create dataset only with tumor images for VAE to learn 'tumor' distribution
    tumor_images = images[labels == 1]
    return images, labels, tumor_images

path = "./brain_tumor_dataset"
all_images, labels, tumor_images = load_data(path)

# Let's train the VAE specifically to understand brain structural variations
# and what it looks like with/without tumor. We train on ALL images for the latent walk
x_train = all_images

# 2. VAE Architecture
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
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
# For 224 inputs, passing through 5 stride-2 convs gets to 7x7x512
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

# VAE Class
class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
            
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

# 3. Train Model
vae = VAE(encoder, decoder)
vae.compile(optimizer=tf.keras.optimizers.Adam())
print("Training VAE...")
history = vae.fit(x_train, epochs=epochs, batch_size=batch_size)

# PLOT LOSS
plt.figure(figsize=(10, 6))
plt.plot(np.arange(1, epochs + 1), history.history["loss"], label="Total Loss")
plt.plot(np.arange(1, epochs + 1), history.history["reconstruction_loss"], label="Reconstruction Loss")
plt.plot(np.arange(1, epochs + 1), history.history["kl_loss"], label="KL Loss")
plt.title("VAE Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig("image/vae_loss_history.jpg")
print("VAE Loss history saved to image/vae_loss_history.jpg")

# SAVE WEIGHTS
encoder.save_weights("vae_encoder_weights.h5")
decoder.save_weights("vae_decoder_weights.h5")
print("Saved encoder and decoder weights to vae_encoder_weights.h5 and vae_decoder_weights.h5")

# 4. Latent Space Manipulation Experiment
print("Generating Latent Space Walk...")
# Generate a grid combining two random dimensions of the latent space
n = 5
import matplotlib.pyplot as plt

# Take base vectors from a tumor image and a non-tumor image
idx_yes = np.where(labels == 1)[0][0]
idx_no = np.where(labels == 0)[0][0]

z_mean_yes, _, _ = encoder.predict(all_images[idx_yes:idx_yes+1])
z_mean_no, _, _ = encoder.predict(all_images[idx_no:idx_no+1])

# We interpolate linearly between the non-tumor encoding to the tumor encoding
alphas = np.linspace(0, 1, n)
fig, axes = plt.subplots(1, n, figsize=(15, 3))

for i, alpha in enumerate(alphas):
    z_interp = z_mean_no * (1 - alpha) + z_mean_yes * alpha
    decoded_img = decoder.predict(z_interp)
    axes[i].imshow(decoded_img[0])
    axes[i].axis("off")
    axes[i].set_title(f"Alpha: {alpha:.2f}")

plt.suptitle("Interpolation from No Tumor to Tumor")
plt.tight_layout()
plt.savefig("image/vae_latent_walk.jpg")
print("Latent walk saved to image/vae_latent_walk.jpg")

# 5. Tumor Classifier Maximization (Latent Space)
print("Finding Latent Vector to Maximize Classifier Tumor Output...")

# Load our base model exactly as defined in the training script
base_model = VGG16(weights='imagenet', input_tensor=layers.Input(shape=(224, 224, 3)), include_top=False)
base_input = base_model.input
base_output = base_model.output
base_output = layers.AveragePooling2D(pool_size=(4, 4))(base_output)
base_output = layers.Flatten(name="flatten")(base_output)
base_output = layers.Dense(64, activation="relu")(base_output)
base_output = layers.Dropout(0.5)(base_output)
base_output = layers.Dense(2, activation="softmax")(base_output)

classifier_model = models.Model(inputs=base_input, outputs=base_output)
classifier_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), metrics=['accuracy'], loss='binary_crossentropy')

# For this demo we load the pre-trained weights if they exist (requires actual training otherwise defaults to init)
# We will optimize a latent vector 'z' starting from normal distribution
loss_z = tf.Variable(tf.random.normal((1, latent_dim)))

# Gradient ascent loop
print("Running gradient ascent on Latent Space for 100 steps...")
for step in range(100):
    with tf.GradientTape() as tape:
        tape.watch(loss_z)
        
        # 1. Decode the current latent vector to an image
        generated_img = decoder(loss_z)
        
        # 2. VGG16 Classifier expects inputs 0-255 scaled for basic predict, but our image is 0-1 sigmoid
        # Our initial model above used / 255 normalized inputs to VGG16 (if we trained it that way).
        # We assume the classifier output takes 0-1 inputs directly here as we did in feature.
        prediction = classifier_model(generated_img)
        
        # Class index 1 is assumed to be tumor ('yes')
        loss = prediction[:, 1]
        
    # Get gradients of the loss with respect to the LATENT vector z
    grads = tape.gradient(loss, loss_z)
    
    # Normalize gradients
    grads = tf.math.l2_normalize(grads)
    
    # Apply gradients (Ascent)
    loss_z.assign_add(0.1 * grads)

# Once optimized, get the final image
maximized_img = decoder(loss_z)
maximized_img = tf.squeeze(maximized_img).numpy()

# the decoder outputs values 0-1 (due to sigmoid), so it's ready to show
plt.figure(figsize=(5, 5))
plt.imshow(maximized_img)
plt.axis("off")
plt.title("Image that Maximizes Tumor Classification (from Latent Vector)")
plt.tight_layout()
plt.savefig("image/tumor_maximization.jpg")
print("Tumor maximization image saved to image/tumor_maximization.jpg")

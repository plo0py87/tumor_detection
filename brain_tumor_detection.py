#!/usr/bin/env python
# coding: utf-8

# ## Brain Tumor Detection

# In[1]:


# Import necessary modules

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from imutils import paths
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import cv2


# In[2]:


# Load the images directories
path = "./brain_tumor_dataset"
print(os.listdir(path))

image_paths = list(paths.list_images(path))
print(len(image_paths))


# In[3]:


# 
images = []
labels = []

for image_path in image_paths:
    label = image_path.split(os.path.sep)[-2]
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))

    images.append(image)
    labels.append(label)


# In[4]:


# Plot an image
def plot_image(image):
    plt.imshow(image)

plot_image(images[0])


# In[5]:


# Convert into numpy arrays
images = np.array(images) / 255.0
labels = np.array(labels)


# In[6]:


# Perform One-hot encoding
label_binarizer = LabelBinarizer()
labels = label_binarizer.fit_transform(labels)
labels = to_categorical(labels)

print(labels[0])


# In[7]:


#Split the dataset
(train_X, test_X, train_Y, test_Y) = train_test_split(images, labels, test_size= 0.10, random_state= 42, stratify= labels)


# In[8]:


# Build the Image Data Generator
train_generator = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True, # 大腦是對稱的，翻轉是合理的
    fill_mode='nearest'
)


# In[9]:


# Build the model
base_model = VGG16(weights= 'imagenet', input_tensor= Input(shape = (224, 224, 3)), include_top= False)
base_input = base_model.input
base_output = base_model.output
base_output = AveragePooling2D(pool_size=(4, 4))(base_output)
base_output = Flatten(name="flatten")(base_output)
base_output = Dense(64, activation="relu")(base_output)
base_output = Dropout(0.5)(base_output)
base_output = Dense(2, activation="softmax")(base_output)


# In[10]:


# Freeze the layers
for layer in base_model.layers:
    layer.trainable = False


# In[11]:


# Compile the model
model = Model(inputs = base_input, outputs = base_output)
model.compile(optimizer= Adam(learning_rate= 1e-3), metrics= ['accuracy'], loss= 'binary_crossentropy')


# In[12]:


# Let's see the architecture summary of our model
model.summary()


# In[13]:


batch_size = 8
train_steps = len(train_X) // batch_size
validation_steps = len(test_X) // batch_size
epochs = 60


# In[14]:


# Fit the model
history = model.fit_generator(train_generator.flow(train_X, train_Y, batch_size= batch_size),
                              steps_per_epoch= train_steps,
                              validation_data = (test_X, test_Y),
                              validation_steps= validation_steps,
                              epochs= epochs)


# In[15]:


# Evaluate the model
predictions = model.predict(test_X, batch_size= batch_size)
predictions = np.argmax(predictions, axis= 1)
actuals = np.argmax(test_Y, axis= 1)


# In[16]:


# Print Classification report and Confusion matrix
print(classification_report(actuals, predictions, target_names= label_binarizer.classes_))

cm = confusion_matrix(actuals, predictions)
print(cm)


# In[17]:


# Final accuracy of our model
total = sum(sum(cm))
accuracy = (cm[0, 0] + cm[1, 1]) / total
print("Accuracy: {:.4f}".format(accuracy))


# In[18]:


# Plot the losses and accuracies
N = epochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), history.history["loss"], label= "train_loss")
plt.plot(np.arange(0, N), history.history["val_loss"], label= "val_loss")

plt.plot(np.arange(0, N), history.history["accuracy"], label= "train_acc")
plt.plot(np.arange(0, N), history.history["val_accuracy"], label= "val_acc")

plt.title("Training Loss and Accuracy on Brain Dataset")
plt.xlabel("Epoch")
plt.ylabel("Loss / Accuracy")
plt.legend(loc= "lower left")
plt.savefig("image/plot.jpg")


# In[ ]:

# ----------------- GRAD-CAM Visualization -----------------
import tensorflow as tf

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img, heatmap, cam_path="image/saliency_map_result.jpg", alpha=0.4):
    heatmap = np.uint8(255 * heatmap)
    jet = plt.cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    
    jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.utils.img_to_array(jet_heatmap)
    
    superimposed_img = jet_heatmap * alpha + img * (1 - alpha)
    superimposed_img = tf.keras.utils.array_to_img(superimposed_img)
    superimposed_img.save(cam_path)
    print(f"Saliency map saved to: {cam_path}")

# Select 3 positive test images randomly from test_X if possible, aiming for a tumor case
target_count = 3
count = 0
plt.figure(figsize=(15, 6))

for i in range(len(actuals)):
    # class 1 is usually 'yes' tumor if label_binarizer classes are sorted
    if actuals[i] == 1 and predictions[i] == 1:
        img_array = test_X[i:i+1]
        pred = predictions[i]
        actual = actuals[i]
        
        print(f"Generating Grad-CAM for Test Image {i} - Actual: {actual}, Predicted: {pred}")
        heatmap = make_gradcam_heatmap(img_array, model, 'block5_conv3', pred_index=pred)
        
        # Get unnormalized image for visualization
        orig_img = test_X[i] * 255.0
        
        heatmap_uint8 = np.uint8(255 * heatmap)
        jet = plt.cm.get_cmap("jet")
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap_uint8]
        
        jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((orig_img.shape[1], orig_img.shape[0]))
        jet_heatmap = tf.keras.utils.img_to_array(jet_heatmap)
        
        superimposed_img = jet_heatmap * 0.4 + orig_img * 0.6
        superimposed_img = tf.keras.utils.array_to_img(superimposed_img)
        
        plt.subplot(2, target_count, count + 1)
        plt.imshow(tf.keras.utils.array_to_img(orig_img))
        plt.title(f"Original {count+1}")
        plt.axis('off')
        
        plt.subplot(2, target_count, count + target_count + 1)
        plt.imshow(superimposed_img)
        plt.title(f"Grad-CAM {count+1}")
        plt.axis('off')
        
        count += 1
        if count >= target_count:
            break

plt.tight_layout()
plt.savefig("image/gradcam_results.jpg")
print("Saliency maps saved to: image/gradcam_results.jpg")

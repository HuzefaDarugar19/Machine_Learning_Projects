🧠 Intel Image Classification with CNN (Built from Scratch)
This project classifies natural scenes using a Convolutional Neural Network (CNN) built from scratch with TensorFlow/Keras. It leverages data augmentation and categorical cross-entropy to train the model on the popular Intel Image Classification Dataset.

📁 Dataset Overview
Dataset: Intel Image Classification (Kaggle)

Categories:

buildings, forest, glacier, mountain, sea, street

Train path used: /kaggle/input/intel-image-classification/seg_train/seg_train

🚀 Model Pipeline Summary
Load images using ImageDataGenerator with augmentation

Build a CNN model using Conv2D, MaxPooling2D, Flatten, Dense, Dropout

Train using categorical_crossentropy

Predict and visualize one image from validation set

📦 Libraries Used
bash
tensorflow
numpy
matplotlib
🧪 Code Walkthrough
🔄 Data Augmentation & Loading
python

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=30,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

path = '/kaggle/input/intel-image-classification/seg_train/seg_train'

train_gen = train_datagen.flow_from_directory(
    path,
    target_size=(150,150),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

test_gen = train_datagen.flow_from_directory(
    path,
    target_size=(150,150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)
🧠 CNN Architecture
python
Copy
Edit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(6, activation='softmax')  # 6 classes
])
⚙️ Compile & Train
python
Copy
Edit
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_gen, validation_data=test_gen, epochs=4)
🔍 Make Prediction & Visualize
python
Copy
Edit
import numpy as np
import matplotlib.pyplot as plt

# Get a batch from test generator
images, labels = test_gen[0]

# Predict the 13th image in the batch
pred = model.predict(np.expand_dims(images[12], axis=0))
pred_class = np.argmax(pred)

# Map predicted index to class name
class_names = list(test_gen.class_indices.keys())
print(class_names[pred_class])  # e.g., "mountain"

# Display the image and label
plt.imshow(images[12])
print("Actual Label (one-hot):", labels[12])
📊 Output Example
text
Copy
Edit
Predicted: mountain
Actual (one-hot): [0. 0. 0. 1. 0. 0.]
📌 Notes
The model uses 3 Conv-Pool blocks, followed by Flatten → Dense → Dropout → Output.

Training is lightweight and completes in a few epochs (on GPU).

Uses softmax for multi-class classification and categorical crossentropy loss.

✨ Future Enhancements
Add EarlyStopping, ModelCheckpoint

Try other optimizers like RMSProp, SGD

Evaluate confusion matrix & classification report

Add GUI for image prediction

🙌 Author
Huzefa Darugar
Aspiring Data Scientist • DL + Power BI Enthusiast
This project is the staring of deep learning projects by me from scratch which shows deep foundational knowledge in
the field of AI .

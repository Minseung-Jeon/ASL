# Import necessary libraries
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json

# Config TensorFlow to use GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Set memory growth to true to avoid memory allocation issues
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU is available and configured for training.")
    except RuntimeError as e:
        print(f"Error configuring GPU: {e}")
else:
    print("GPU is not available. Training will use CPU.")

# Parameters
dataset_path = 'E:/OneDrive/SCHOOL/RYERSON/COURSES/CPS (Computer Science)/CPS843-Into to Computer Vision/project/asl-alphabet/asl_alphabet_train/asl_alphabet_train'  # Update path
img_height, img_width = 200, 200  # Image dimensions
batch_size = 32
epochs = 30  # Increased for better training
early_stopping_patience = 5  # Early stopping patience

# Load and Preprocess Dataset
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # Normalize pixel values to [0, 1]
    rotation_range=20,  # Augmentation: Rotate images
    width_shift_range=0.2,  # Augmentation: Horizontal shift
    height_shift_range=0.2,  # Augmentation: Vertical shift
    shear_range=0.15,  # Augmentation: Shear
    zoom_range=0.2,  # Augmentation: Zoom
    horizontal_flip=True,  # Augmentation: Horizontal flip
    validation_split=0.2  # Use 20% of data for validation
)

train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'  # Set as training data
)

validation_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'  # Set as validation data
)

# Define a more complex CNN model
model = models.Sequential([
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),  # Add dropout to reduce overfitting
    layers.Dense(29, activation='softmax')  # 26 letters + SPACE, DELETE, NOTHING
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Reduced learning rate for better training
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Add callbacks for early stopping and saving the best model
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=early_stopping_patience, restore_best_weights=True)
model_checkpoint = callbacks.ModelCheckpoint("asl_alphabet_model.keras", save_best_only=True)

# Train the model with added callbacks
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=epochs,
    callbacks=[early_stopping, model_checkpoint]
)

# Save training history to a JSON file
with open("training_history.json", "w") as f:
    json.dump(history.history, f)

# Save the final model
model.save("final_asl_alphabet_model.h5")
print("Model and training history saved.")

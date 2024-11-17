#import tensorflow library
import tensorflow as tf
#imports ImageDataGenerator class from Keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#creating ImageDataGenerator object
train_datagen = ImageDataGenerator(
    #rescales pixel values to 0-1 range from 0-255 range
    rescale = 1. /255,
    rotation_range = 20,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    #shearing the image for variations 
    shear_range = 0.2,
    #zoom transformation for variations
    zoom_range = 0.2,
    #randomly flips the images horizontally, helps prevent overfitting
    #also helps doubles the size of training dataset, providing more examples to learn from
    horizontal_flip = True,
    #20% of the training data will be used as validation data
    validation_split = 0.2,
    fill_mode = 'nearest'
)

#Creates a data generator that loads images from the specified directory
train_generator = train_datagen.flow_from_directory(
    r'archive\asl_alphabet_train\asl_alphabet_train',
    # resizes images to 64x64 pixels
    target_size = (128, 128),
    #specifies the batch size for training
    batch_size = 32,
    #indicates categorical classification
    class_mode = 'categorical',
    #Loads images in grayscale mode
    color_mode = 'grayscale',
    subset = 'training',
)

validation_generator = train_datagen.flow_from_directory(
    r'archive\asl_alphabet_train\asl_alphabet_train',
    target_size = (128, 128),
    batch_size = 32,
    class_mode = 'categorical',
    color_mode = 'grayscale',
    subset = 'validation',
)

#defining CNN model using tensorflow and keras
#creates a sequential model where layers are linearly stacked
model = tf.keras.models.Sequential([
    #takes 64x64 pixel grayscale images as input
    tf.keras.layers.Input(shape = (128, 128, 1)),
    #adds convolutional layers to extract features
    #32 filter(kernals) in the layer
    # size of the filter is 3x3
    # uses relu activation function
    tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (64, 64, 1)),
    #adds pooling layer to reduce dimensionality 
    #size of each filter is 2x2
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    # this layer converts the multi-dimensional data into a single, long vector
    tf.keras.layers.Flatten(),

    #dense layer where each neuron is connected to every neuron in the previous layer
    #first one has 128 neurons, second one has 29 neurons
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dense(29, activation='softmax')
])

#compiles the model
#uses adam optimizer
#loss function is categorical crossentropy (measures the difference between model's predictions and true label)
#specifies the metric used to evaluate the mode's performance during training
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#initializes training of the model (actual learning happens)
#train generator is the data generator created earlier
#model goes through the training data 20 times
model.fit(
    train_generator,
    epochs=15,
    validation_data = validation_generator
)

model.save('asl_model.h5')

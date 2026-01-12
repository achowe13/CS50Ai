import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    cwd = os.path.join(os.getcwd(), data_dir)
    images = []
    labels = []
    
    for root, sub, files in os.walk(cwd):
        for file in files:
            file_path = os.path.join(root, file)
            image = cv2.imread(file_path)
                
            if image is not None:
                resized_image = (cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))) / 255
                np_image = np.array(resized_image)
                images.append(np_image)
                folder = os.path.basename(root) 
                labels.append(int(folder))
    
    return((images, labels))

def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    #1: Relu same as example--after 10: - 2ms/step - accuracy 0.9303 loss 0.2198, final: accuracy 0.9769 loss 0.1112
    #model = tf.keras.models.Sequential([
    #    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    #    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    #    tf.keras.layers.Flatten(),
    #    tf.keras.layers.Dense(128, activation="relu"),
    #    tf.keras.layers.Dropout(0.5),
    #    tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    #])
    
    #2: Relu same as example--after 10: - 2ms/step - accuracy 0.9470 loss 0.1562, final: accuracy 0.9860 loss 0.0656
    #model = tf.keras.models.Sequential([
    #    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    #    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    #    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    #    tf.keras.layers.Flatten(),
    #    tf.keras.layers.Dense(128, activation="relu"),
    #    tf.keras.layers.Dropout(0.5),
    #    tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    #])
    
    #2: Leaky Relu--after 10: - 2ms/step - accuracy 0.9604 loss 0.1388, final: accuracy 0.9767 loss 0.1068
    #model = tf.keras.models.Sequential([
    #    tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.01), input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    #    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    #    tf.keras.layers.Flatten(),
    #    tf.keras.layers.Dense(128, activation=tf.keras.layers.LeakyReLU(alpha=0.01)),
    #    tf.keras.layers.Dropout(0.5),
    #    tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    #])
    
    #3: ELU--after 10: - 2ms/step - accuracy 0.9710 loss 0.0974, final: accuracy 0.9740 loss 0.1053
    #model = tf.keras.models.Sequential([
    #    tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.layers.ELU(alpha=1.0), input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    #    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    #    tf.keras.layers.Flatten(),
    #    tf.keras.layers.Dense(128, activation=tf.keras.layers.ELU(alpha=1.0)),
    #    tf.keras.layers.Dropout(0.5),
    #    tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    #])
    
    #4: Relu with normalization--after 10: - 2ms/step - accuracy 0.9544 loss 0.1465, final: accuracy 0.9736 loss 0.1184
    #model = tf.keras.models.Sequential([
    #    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    #    tf.keras.layers.BatchNormalization(),
    #    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    #    tf.keras.layers.Flatten(),
    #    tf.keras.layers.Dense(128, activation="relu"),
    #    tf.keras.layers.Dropout(0.5),
    #    tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    #])
    
    #5: Leaky Relu with normalization--after 10: - 2ms/step - accuracy 0.9547 loss 0.1411, final: accuracy 0.9753 loss 0.0943
    #model = tf.keras.models.Sequential([
    #    tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.01), input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    #    tf.keras.layers.BatchNormalization(),
    #    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    #    tf.keras.layers.Flatten(),
    #    tf.keras.layers.Dense(128, activation=tf.keras.layers.LeakyReLU(alpha=0.01)),
    #    tf.keras.layers.Dropout(0.5),
    #    tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    #])
    
    #6: ELU with normalization--after 10: - 2ms/step - accuracy 0.9573 loss 0.1469, final: accuracy 0.9658 loss 0.1474
    #model = tf.keras.models.Sequential([
    #    tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.layers.ELU(alpha=1.0), input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    #    tf.keras.layers.BatchNormalization(),
    #    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    #    tf.keras.layers.Flatten(),
    #    tf.keras.layers.Dense(128, activation=tf.keras.layers.ELU(alpha=1.0)),
    #    tf.keras.layers.Dropout(0.5),
    #    tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    #])
    
    #2: Relu 2 convolution layers with normalization **this took a long time to train (18ms/step)--after 10: - 3ms/step - accuracy 0.9761 loss 0.0773, final: accuracy 0.9894 loss 0.0636
    #model = tf.keras.models.Sequential([
    #    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    #    tf.keras.layers.BatchNormalization(),
    #    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    #    tf.keras.layers.BatchNormalization(),
    #    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    #    tf.keras.layers.Flatten(),
    #    tf.keras.layers.Dense(128, activation="relu"),
    #    tf.keras.layers.Dropout(0.5),
    #    tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    #])
    
    #4: 16 kernals   after 10: - 5ms/step - accuracy: 0.9389 loss: 0.1942 final: accuracy: 0.9721 loss: 0.1185,
    #model = tf.keras.models.Sequential([
    #    tf.keras.layers.Conv2D(16, (3, 3), activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    #    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    #    tf.keras.layers.Flatten(),
    #    tf.keras.layers.Dense(256, activation="relu"),
    #    tf.keras.layers.Dropout(0.5),
    #    tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    #])
    
    #4: 64 kernals    after 10: - 5ms/step - accuracy: 0.9624 loss: 0.1205 final: accuracy: 0.9782 loss: 0.0933,
    #model = tf.keras.models.Sequential([
    #    tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    #    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    #    tf.keras.layers.Flatten(),
    #    tf.keras.layers.Dense(256, activation="relu"),
    #    tf.keras.layers.Dropout(0.5),
    #    tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    #])
    
    #4: 128 kernals    after 10: accuracy: 0.9564 loss: 0.1440  final: 5ms/step - accuracy: 0.9768 loss: 0.1029,
    #model = tf.keras.models.Sequential([
    #    tf.keras.layers.Conv2D(128, (3, 3), activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    #    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    #    tf.keras.layers.Flatten(),
    #    tf.keras.layers.Dense(256, activation="relu"),
    #    tf.keras.layers.Dropout(0.5),
    #    tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    #])
    
    #4: 16 and 32    after 10: - 5ms/step - accuracy: 0.9840 loss: 0.0541  final: accuracy: 0.9907 loss: 0.0467,
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])
    
    #4: after 10: - 5ms/step - accuracy: 0.9868 loss: 0.0441 final: accuracy: 0.9940 loss: 0.0300,
    #model = tf.keras.models.Sequential([
    #    tf.keras.layers.Conv2D(16, (3, 3), activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    #    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    #    tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    #    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    #    tf.keras.layers.Flatten(),
    #    tf.keras.layers.Dense(256, activation="relu"),
    #    tf.keras.layers.Dropout(0.5),
    #    tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    #])
    
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model


if __name__ == "__main__":
    main()

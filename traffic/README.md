the first attempt with the ai was just using the provided example from the class notes to get a feel for how to use the program.

    #1: Relu same as example--after 10: accuracy 0.9303 loss 0.2198, final: 2ms/step - accuracy 0.9769 loss 0.1112
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

After the demo the first thing i tried was to add an extra convolution layer which increased the final accuracy by a little less than 1% but almost doubled the training step time and increased the final step by 50%.
 
    #2: Relu same as example--after 10: accuracy 0.9470 loss 0.1562, final: 2ms/step - accuracy 0.9860 loss 0.0656
        x2 - tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

From there I tried a couple of different activations (Leaky Relu and ELU) to see if any of them made much of a difference in the accuracy. After testing I found that Leaky Relu seemed to provide betterresults than just the Relu and ELU was out performing both of them in the final accuracy tests and the time per step was the same as the original Relu activation.

    #3: Leaky Relu--after 10: accuracy 0.9604 loss 0.1388, final: 2ms/step - accuracy 0.9767 loss 0.1068
        tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.01), input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

    #4: ELU--after 10: accuracy 0.9710 loss 0.0974, final: 2ms/step - accuracy 0.9740 loss 0.1053
        tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.layers.ELU(alpha=1.0), input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

After the activation changes I had read about batch normalization so i tried adding it to the original example with the single convolution layer and the Relu activation as well as tried on the other 2 activation types.

    #5: Relu with normalization--after 10: accuracy 0.9544 loss 0.1465, final: 2ms/step - accuracy 0.9736 loss 0.1184
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ]) 

    #6: Leaky Relu with normalization--after 10: accuracy 0.9547 loss 0.1411, final: 2ms/step - accuracy 0.9753 loss 0.0943
        tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.01), input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        tf.keras.layers.BatchNormalization(),
    
    #7: ELU with normalization--after 10: accuracy 0.9573 loss 0.1469, final: 2ms/step - accuracy 0.9658 loss 0.1474   
        tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.layers.ELU(alpha=1.0), input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        tf.keras.layers.BatchNormalization(),

The result of these tests gave a faster initial increase in accuracy durring training but led to a slight decrease in the final accuracy.

after this attempt I decided to try out the batch normalization with extra convolution layers.

    #8: Relu 2 convolution layers with normalization (training 18ms/step)--after 10: accuracy 0.9761 loss 0.0773, final: 3ms/step - accuracy 0.9894 loss 0.0636
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        tf.keras.layers.BatchNormalization(),

this led to the hichest accuracy that i had seen so far at 98.94% but also a very long traing step that was nearly 4x the original with a 50% increase to the step time in the final.  The next change that I made was changing the amount of kernals in the convolution layer. I tried 64 which had better training accuracy and slightly better final accuracy for a tradeoff of a 5ms final and a 26ms training step time:

    #9: 64 kernals    after 10: accuracy: 0.9624 loss: 0.1205 final: 5ms/step - accuracy: 0.9782 loss: 0.0933,
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),

next was 128 kernals which actually lowered the final accuracy and while the final step was the same as with 64 kernals the training time was again doubled to 54ms/step:

    #10: 128 kernals    after 10: accuracy: 0.9564 loss: 0.1440  final: 5ms/step - accuracy: 0.9768 loss: 0.1029,
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),

and 16 which was definitley faster, however the accuracy suffered because of it with a lower after 10 accuracy as well as a lower final accuracy:

    #11: 16 kernals   after 10: accuracy: 0.9389 loss: 0.1942 final: 5ms/step - accuracy: 0.9721 loss: 0.1185,
        tf.keras.layers.Conv2D(16, (3, 3), activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),

finally the last change was to add multiple convolution layers with increasing amounts of kernals:

    #12: 16 and 32    after 10: accuracy: 0.9840 loss: 0.0541  final: 5ms/step - accuracy: 0.9907 loss: 0.0467,
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])
    
    #13: 16, 32 and 64    after 10: accuracy: 0.9868 loss: 0.0441 final: 5ms/step -  accuracy: 0.9940 loss: 0.0300,
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

this final change resulted in the best accuracy of the day with the 3 convolution layer model having a final accuracy of 99.4% with only 0.0300 loss.
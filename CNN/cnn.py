# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
'''
Sequential -- To initialize a NN, a sequence of layers.
Conv2D - For the convolutional layers, since images -- 2D images, hence, 2D.
MaxPooling2D - for Pooling layers.
Faltten - for Flattening layers.
Dense - Add the Fully Connected layers in the ANN.
'''

# Initialising the CNN
classifier = Sequential()

# Step 1 - Add Convolution Layers
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
'''
# Conv2D( 
-- number of filters, (number of rows of the filter, number of the columns of the filter), start with 32 in the first layer, then do 64, then 128 in the second, and so on!
-- input_shape -- shape of the input image, need to convert all our input images into one same size. (rows x columns x depth)
-- activation function - relu)
'''

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
'''
Just have one feature -- Pooling Layer size -- we want a 2 x 2 pooling layer, and a stride of 

Reduce the number of nodes in the Fully Connected layer in the future.
'''

# Adding a second convolutional layer
# Making the network deeper, because first we tried with onyl 1 layer of convolution and 1 layer of maxpool,
# but then, we got 75% test set accuracy and 84% training set accuracy, so, overfitting, hence, made the network deeper.
'''
Have three options - 
1. Add another Convolutional and Maxpool layer.
2. Add another Fully Connected layer.
3. Add both

No need to add the input_shape parameter, bcs Keras will know that the input image will be 64 x 64 x 3, bcs of the previous layer.
- Could have made the number of feature detectors to 64, good practice, but just doing with 32 filters/feature detectors.
'''
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())


# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# optimizer = stochastic gradient descent -- when should the weights be updated.
# if more than 2 output variables, we would need to choose -- categorical_crossentropy



#/////////////////////////////////////////////////////FITTING CNN TO THE IMAGES////////////////////////////////////////////////////
# Part 2 - Fitting the CNN to the images

'''
Image Augmetation -- Prevent Overfitting the images on train set, hence some data preprocessing needed.
- Will create many batches of our images and in each batch it will apply random transformations, rotation, flipping, shifting, so many more
diverse images in side the batches, and a lot more material to train. -- augmenting the number of  images in our dataset.
And since these are random transformations, model will never find the same images again. -- reduces overfitting.

'''

from keras.preprocessing.image import ImageDataGenerator

'''
ImageDataGenerator makes the Image augmentation.
- Rescale part - it is the feature scaling step, like we had in the normal ANN, rescale all pixel values from 0 to 1.
- Vertical flip also there, but not used because we cant have dogs and cats upside down.
'''
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)


#for the test set, only rescaling ths images, because image augmentation is done only for the train part.
test_datagen = ImageDataGenerator(rescale = 1./255)


# flow from directory, because we have segregated the images in different folders which act as filters.
# creates the training set
# 'dataset/training_set -- gives the path of the training set, which has two folders as dogs and cats'
# target_size = 64 x 64, because that is the size of the input images that we defined in the CNN above.
# classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu')).. this is 64 x 64
# class_mode -- tells if the dependant variable has two categories or more than 2 categories.
'''
The project structure is as follows:

- script.py
- dataset
    - training_set
        - cats
        - dogs
    - test_set
        - cats
        - dogs
'''
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

#creates the test set.
# images of our test set will also have 64 x 64 images, hence, target_size = 64 x 64
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

#this will fit the CNN in the training set, and test its performace on the test set.
# steps_per_epoch -- number of images in the training set.
classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000)
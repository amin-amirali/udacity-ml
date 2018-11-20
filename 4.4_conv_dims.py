from keras.models import Sequential
from keras.layers import Conv2D

# filters - The number of filters.
# kernel_size - Number specifying both the height and width of the (square) convolution window.
# Optional arguments:
# strides - The stride of the convolution. If you don't specify anything, strides is set to 1.
# padding - One of 'valid' or 'same'. If you don't specify anything, padding is set to 'valid'.
# activation - Typically 'relu'. If you don't specify anything, no activation is applied. You are strongly encouraged to add a ReLU activation function to every convolutional layer in your networks.

# model = Sequential()
# model.add(Conv2D(filters=18, kernel_size=2, strides=2, padding='same', 
#     activation='relu', input_shape=(200, 200, 1)))
# model.summary()

from keras.models import Sequential
from keras.layers import Conv2D

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=3, strides=2, padding='same', 
    activation='relu', input_shape=(128, 128, 3)))
model.summary()

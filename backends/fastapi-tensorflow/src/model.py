from keras import optimizers
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.models import Sequential


class CNN(Sequential):
    def __init__(self):
        super().__init__()

        self.add(Conv2D(10, kernel_size=5, input_shape=(28, 28, 1)))
        self.add(MaxPooling2D(pool_size=(2, 2)))
        self.add(Conv2D(20, kernel_size=5))
        self.add(MaxPooling2D(pool_size=(2, 2)))
        self.add(Dropout(0.5))
        self.add(Flatten())
        self.add(Dense(50, activation='relu'))
        self.add(Dropout(0.5))
        self.add(Dense(10, activation='softmax'))

        self.optimizer = optimizers.Adam(learning_rate=0.001)
        self.compile(optimizer=self.optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

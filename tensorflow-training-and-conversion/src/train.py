from datetime import datetime
from os import makedirs
from os.path import join as path_join

from keras import optimizers

from .callbacks import SaveModelPerEpochCallback
from .config import config, models_dir
from .dataset import data_loaders
from .model import CNN


def train(models_save_dir, num_epochs):
    model = CNN()
    optimizer = optimizers.Adam(lr=0.001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    x_train, y_train = data_loaders['train']

    save_model_per_epoch_callback = SaveModelPerEpochCallback(models_save_dir)

    model.fit(x_train, y_train, epochs=num_epochs, callbacks=[save_model_per_epoch_callback])


if __name__ == "__main__":
    training_id = "tensorflow---" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")    
    models_save_dir = path_join(models_dir, training_id)
    makedirs(models_save_dir)

    train(models_save_dir, config['training_num_epochs'])

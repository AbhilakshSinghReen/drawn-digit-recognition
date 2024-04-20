from os.path import join as path_join

from keras import callbacks


class SaveModelPerEpochCallback(callbacks.Callback):
    def __init__(self, models_save_dir):
        super().__init__()
        self.models_save_dir = models_save_dir

    def on_epoch_end(self, epoch, logs=None):
        model_save_file_path = path_join(self.models_save_dir, f"epoch-{epoch}.model")
        self.model.save(model_save_file_path)

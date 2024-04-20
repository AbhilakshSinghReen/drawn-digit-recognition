from os.path import dirname, join as path_join
from yaml import FullLoader as yaml_FullLoader, load as yaml_load


config_file_path = path_join(dirname(dirname(__file__)), "config.yaml")
with open(config_file_path, "r") as model_params_file:
    config = yaml_load(model_params_file, Loader=yaml_FullLoader)

models_dir = path_join(dirname(dirname(dirname(__file__))), "models")

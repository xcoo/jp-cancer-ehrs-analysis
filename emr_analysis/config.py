import yaml


def load_config(path):
    with open(path) as config_yml:
        config = yaml.load(config_yml, Loader=yaml.FullLoader)
    return config

from model.SwinCD import SwinCD


def build_model(config):
    model = SwinCD(config)
    return model


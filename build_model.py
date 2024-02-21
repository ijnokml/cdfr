from model.SwinUperNet import SwinUperNet


def build_model(config):
    model = SwinUperNet(config)
    return model


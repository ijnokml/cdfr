from test import validate
import torch

# Other

from utils import load_checkpoint
from build_model import build_model

def test(config, data_loader_test, val_loader):

    model = build_model(config)
    model.cuda()
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of params: {n_parameters}")

    load_checkpoint(config=config, model=model)


    validate(model, data_loader_test, None)

    return 0



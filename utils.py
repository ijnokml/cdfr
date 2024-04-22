import os
import torch
import os
try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def load_checkpoint(config, model, optimizer, lr_scheduler):
    print("==============> Resuming form {}....................".format(config.MODEL.RESUME))
    checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    if 'optimizer_state_dict' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        config.freeze()
        if 'amp' in checkpoint and config.SYSTEM.AMP_OPT_LEVEL != "O0" and checkpoint['config'].SYSTEM.AMP_OPT_LEVEL != "O0":
            amp.load_state_dict(checkpoint['amp'])

    del checkpoint
    torch.cuda.empty_cache()
    return 0

def save_checkpoint(config, epoch, model, optimizer, lr_scheduler):
    save_state = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'epoch': epoch,
                  'config': config}
    if config.SYSTEM.AMP_OPT_LEVEL != "O0":
        save_state['amp'] = amp.state_dict()

    save_path = os.path.join(config.SYSTEM.OUTPUT, f'ckpt_epoch_{epoch}.pth')
    torch.save(save_state, save_path)
    print('>>>model on {} epochs saved<<<<'.format(epoch))
    return 0

def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def tensor_len(tensor):
    len = 1
    for i in tensor.size():
        len *= i
    return len

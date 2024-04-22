
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

# Other
from test import validate

from utils import save_checkpoint, load_checkpoint, get_grad_norm
from build_optimizer import build_optimizer
from build_lr_scheduler import build_scheduler
from build_model import build_model
from build_loss import Uper_Loss
from mmcv.cnn.utils import revert_sync_batchnorm

# data
try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def train(config, data_loader_train, val_loader):

    model = build_model(config)
    if config.SYSTEM.LOCAL_RANK != 'cpu':
        model.to(torch.device('cuda', config.SYSTEM.LOCAL_RANK))
    else:
        model.to(torch.device('cpu'))

    # may the lr drop too fast, enlarge 0.95 para
    # if config.TRAIN.OPTIMIZER.NAME == 'adamw':
    # optimizer = torch.optim.AdamW(net.parameters(),) #Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    optimizer = build_optimizer(config=config, model=model)

    if config.SYSTEM.AMP_OPT_LEVEL != 'O0':
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=config.SYSTEM.AMP_OPT_LEVEL)
    if config.SYSTEM.LOCAL_RANK != 'cpu':
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[config.SYSTEM.LOCAL_RANK], broadcast_buffers=False)
        model_without_ddp = model.module
    else:
        model = revert_sync_batchnorm(model)
        model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of params: {n_parameters}")

    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, exp)
    scheduler = build_scheduler(config=config, optimizer=optimizer,
                                n_iter_per_epoch=len(data_loader_train))
    if config.TRAIN.LOSS_WEIGHT is None:
        criterion = Uper_Loss()
    else:
        weihgt_tensor = torch.tensor(config.TRAIN.LOSS_WEIGHT,device=config.SYSTEM.LOCAL_RANK)
        criterion = Uper_Loss(decode_weights=weihgt_tensor, aux_weights=weihgt_tensor)


    if config.MODEL.RESUME:
        msg0 = load_checkpoint(config=config, model=model_without_ddp,
                           optimizer=optimizer, lr_scheduler=scheduler)
    writer = SummaryWriter('./log')  # TODO.format(net_name)

    # start train
    for epoch_index in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        if config.SYSTEM.LOCAL_RANK != 'cpu':
            data_loader_train.sampler.set_epoch(epoch_index)
            # 设置当前的 epoch，为了让不同的结点之间保持同步。

        model.train()
        optimizer.zero_grad()
        num_steps = len(data_loader_train)
        learn_rate = optimizer.state_dict()['param_groups'][0]['lr']
        if config.SYSTEM.LOCAL_RANK != 'cpu':
            print("Local Rank: {}, Epoch: {}, Training ...".format(dist.get_rank(), epoch_index))

        for idx, batch in enumerate(data_loader_train):
            
            if config.SYSTEM.LOCAL_RANK != 'cpu':
                I1 = batch[0][0].to(torch.device('cuda', config.SYSTEM.LOCAL_RANK), non_blocking=True)
                I2 = batch[1][0].to(torch.device('cuda', config.SYSTEM.LOCAL_RANK), non_blocking=True)
                label = torch.squeeze(batch[2].to(torch.device('cuda', config.SYSTEM.LOCAL_RANK), non_blocking=True))
            else:
                I1 = batch[0][0].to(torch.device('cpu'))
                I2 = batch[1][0].to(torch.device('cpu'))
                label = torch.squeeze(batch[2].to(torch.device('cpu')))

            decode_out, aux_out = model(torch.cat([I1, I2],1))
            if config.TRAIN.ACCUMULATION_STEPS > 1:
                loss, _, _ = criterion(decode_out, aux_out, label.long())
                loss = loss / config.TRAIN.ACCUMULATION_STEPS
                if config.SYSTEM.AMP_OPT_LEVEL != "O0":
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    if config.TRAIN.CLIP_GRAD:
                        grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                    else:
                        grad_norm = get_grad_norm(amp.master_params(optimizer))
                else:
                    loss.backward()
                    if config.TRAIN.CLIP_GRAD:
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                    else:
                        grad_norm = get_grad_norm(model.parameters())
                if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step_update(epoch_index * num_steps + idx)
            else:
                loss, _, _ = criterion(decode_out, aux_out, label.long())
                optimizer.zero_grad()
                if config.SYSTEM.AMP_OPT_LEVEL != "O0":
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    if config.TRAIN.CLIP_GRAD:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                    else:
                        grad_norm = get_grad_norm(amp.master_params(optimizer))
                else:
                    loss.backward()
                    if config.TRAIN.CLIP_GRAD:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), config.TRAIN.CLIP_GRAD)
                    else:
                        grad_norm = get_grad_norm(model.parameters())
                optimizer.step()
                scheduler.step_update(epoch_index * num_steps + idx)

            torch.cuda.synchronize()

        if ((epoch_index+1) % config.SYSTEM.PRINT_FREQ) == 0:  # TODO
            validate(config, model, val_loader, criterion,epoch_index,writer)
            writer.add_scalar('learn_rate', learn_rate, epoch_index)
        torch.distributed.barrier()#用于同步进程状态

        if dist.get_rank() == 0 and ((epoch_index+1) % config.SYSTEM.SAVE_FREQ == 0 or epoch_index == (config.TRAIN.EPOCHS - 1)):
            msg0 = save_checkpoint(config=config, epoch=epoch_index, model=model_without_ddp,
                                   optimizer=optimizer, lr_scheduler=scheduler)
        torch.distributed.barrier()#用于同步进程状态

    return 'See Details in Tensorboard'
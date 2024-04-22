import torch
# Other
import numpy as np


@torch.no_grad()
def validate(config, net, dset_loader, criterion,epoch_index,writer):
    net.eval()
    
    total_loss = 0.
    total_loss_decode = 0.
    total_loss_aux = 0.
    if config.SYSTEM.LOCAL_RANK != 'cpu':
        
        total_count = torch.tensor(0, dtype=torch.int64).to(torch.device('cuda', config.SYSTEM.LOCAL_RANK))

        tp = torch.tensor(0, dtype=torch.int64).to(torch.device('cuda', config.SYSTEM.LOCAL_RANK))
        tn = torch.tensor(0, dtype=torch.int64).to(torch.device('cuda', config.SYSTEM.LOCAL_RANK))
        fp = torch.tensor(0, dtype=torch.int64).to(torch.device('cuda', config.SYSTEM.LOCAL_RANK))
        fn = torch.tensor(0, dtype=torch.int64).to(torch.device('cuda', config.SYSTEM.LOCAL_RANK))
    else:
        total_count = torch.tensor(0).to(torch.device('cpu'))
        tp = torch.tensor(0).to(torch.device('cpu'))
        tn = torch.tensor(0).to(torch.device('cpu'))
        fp = torch.tensor(0).to(torch.device('cpu'))
        fn = torch.tensor(0).to(torch.device('cpu'))

# ---------------------------------------

    for testbatch in dset_loader:
        if config.SYSTEM.LOCAL_RANK != 'cpu':
            I1 = testbatch[0][0].to(torch.device('cuda', config.SYSTEM.LOCAL_RANK))
            I2 = testbatch[1][0].to(torch.device('cuda', config.SYSTEM.LOCAL_RANK))
            label = testbatch[2].to(torch.device('cuda', config.SYSTEM.LOCAL_RANK))
        else:
            I1 = testbatch[0][0].to(torch.device('cpu'))
            I2 = testbatch[1][0].to(torch.device('cpu'))
            label = testbatch[2].to(torch.device('cpu'))
        # name = testbatch[3]
        output, aux = net(torch.cat([I1, I2],1))
        loss, decode_loss, aux_loss = criterion(output, aux, label.long())
        pixels = np.prod(label.size())
        total_loss += loss * pixels
        total_loss_decode += decode_loss * pixels
        total_loss_aux += aux_loss * pixels
        total_count += pixels
        _, predicted = torch.max(output, dim=1)
        # c = (predicted.int() == label.int())
        # draw(predicted,name,~c)

        pr = (predicted.int() > 0)
        gt = (label.int() > 0)

        tp += torch.sum(pr & gt)  # np.logical_and(pr, gt).sum()
        # np.logical_and(np.logical_not(pr), np.logical_not(gt)).sum()
        tn += torch.sum((~pr) & (~gt))
        # np.logical_and(pr, np.logical_not(gt)).sum()
        fp += torch.sum(pr & (~gt))
        # np.logical_and(np.logical_not(pr), gt).sum()
        fn += torch.sum((~pr) & gt)
# ---------------------------------------
    net_loss = total_loss/total_count
    net_loss_decode = total_loss_decode/total_count
    net_loss_aux = total_loss_aux/total_count
    net_accuracy = 100 * (tp + tn)/total_count


    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f_meas = 2 * prec * rec / (prec + rec)
    prec_nc = tn / (tn + fn)
    rec_nc = tn / (tn + fp)

    # pr_rec = [prec, rec, f_meas, prec_nc, rec_nc]

    # return net_loss, net_accuracy, class_accuracy, pr_rec

    # print(f'rank:{torch.distributed.get_rank()},netloss:{net_loss}')
    torch.distributed.all_reduce(net_loss, op=torch.distributed.ReduceOp.AVG)
    torch.distributed.all_reduce(net_loss_aux, op=torch.distributed.ReduceOp.AVG)
    torch.distributed.all_reduce(net_loss_decode, op=torch.distributed.ReduceOp.AVG)
    torch.distributed.all_reduce(rec, op=torch.distributed.ReduceOp.AVG)
    torch.distributed.all_reduce(f_meas, op=torch.distributed.ReduceOp.AVG)
    torch.distributed.all_reduce(prec, op=torch.distributed.ReduceOp.AVG)
    torch.distributed.all_reduce(net_accuracy, op=torch.distributed.ReduceOp.AVG)

    torch.distributed.all_reduce(rec_nc, op=torch.distributed.ReduceOp.AVG)
    torch.distributed.all_reduce(prec_nc, op=torch.distributed.ReduceOp.AVG)
    
    
    if torch.distributed.get_rank() == 0:
        writer.add_scalar('test_loss/loss', net_loss, epoch_index)
        writer.add_scalar('test_loss/decode_loss', net_loss_decode, epoch_index)
        writer.add_scalar('test_loss/aux_loss', net_loss_aux, epoch_index)
        writer.add_scalar('test_result/recall', rec, epoch_index)
        writer.add_scalar('test_result/F1', f_meas, epoch_index)
        writer.add_scalar('test_result/precision', prec, epoch_index)

        writer.add_scalar('test_result/recall_nc', rec_nc, epoch_index)

        writer.add_scalar('test_result/precision_nc', prec_nc, epoch_index)
        # writer.add_scalar('test_accurancy/nochange',
        #                 class_accuracy[0], epoch_index)
        # writer.add_scalar('test_accurancy/change',
        #                 class_accuracy[1], epoch_index)
        writer.add_scalar('test_accurancy/accuracy',
                        net_accuracy, epoch_index)
                        

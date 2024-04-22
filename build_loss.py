import torch.nn as nn


class Uper_Loss(nn.Module):
    def __init__(self, decode_weights = None, aux_weights = None, balance_weights = 1) -> None:
        super().__init__()
        self.ce_decode = nn.CrossEntropyLoss(weight=decode_weights)
        self.ce_aux = nn.CrossEntropyLoss(weight=aux_weights)
        self.balance_weights = balance_weights

    def forward(self, decode_outs, aux_outs, label):
        loss = self.ce_decode(decode_outs, label) + self.balance_weights * self.ce_aux(aux_outs, label)
        return loss, self.ce_decode(decode_outs, label), self.balance_weights * self.ce_aux(aux_outs, label)
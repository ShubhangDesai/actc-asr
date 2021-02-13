import torch
import torch.nn.functional as F

import numpy as np

from .CTC import *
from .AbstentionCTC import *

def ctc_loss(y_hat, y, y_hat_lens, target_lens):
    loss = 0.0
    y_hat = y_hat if 'tuple' in str(type(y_hat)) else [y_hat]

    for y_hat_i in y_hat:
        #breakpoint()
        loss_i = F.ctc_loss(y_hat_i.cpu(), y.cpu(), y_hat_lens.cpu(), target_lens.cpu())
        loss += loss_i

    return loss


def get_loss(args):
    if args['loss'] == 'ctc':
        return ctc_loss
    elif args['loss'] == 'ctc_custom':
        return CTC()
    elif args['loss'] == 'abstention':
        return AbstentionCTC()
    else:
        return disagreement_loss

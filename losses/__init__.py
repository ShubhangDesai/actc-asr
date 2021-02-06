import torch
import torch.nn.functional as F

import numpy as np

from .CTC import *
from .AbstentionCTC import *

def ctc_loss(y_hat, y, y_hat_lens, target_lens, w=None):
    loss = 0.0
    y_hat = y_hat if 'tuple' in str(type(y_hat)) else [y_hat]

    for y_hat_i in y_hat:
        loss_i = F.ctc_loss(y_hat_i.cpu(), y.cpu(), y_hat_lens.cpu(), target_lens.cpu(), reduction=('mean' if w is None else 'none'))
        if w is not None: loss = torch.sum(loss_i * w)
        loss += loss_i

    return loss

def disagreement_loss(y_hat, y, y_hat_lens, target_lens, forget_rate, w=None):
    y_hat_1, y_hat_2 = y_hat

    disagree_idx = []
    label_encoder = LabelWorker()
    for i in range(y_hat_1.shape[1]):
        pred_1 = y_hat_1[:, i].data.cpu().numpy()
        pred_2 = y_hat_2[:, i].data.cpu().numpy()

        pred_str_1 = RelaxString(label_encoder.GreedyDecode(pred_1, True))
        pred_str_2 = RelaxString(label_encoder.GreedyDecode(pred_2, True))

        if pred_str_1 != pred_str_2:
            disagree_idx.append(i)
        elif len(pred_str_1.replace(' ', '').replace('.', '').replace(',', '')) <= 1:
            disagree_idx.append(i)

    # TODO: only do this if step < 5000?
    num_remember = y_hat_1.shape[1] if len(disagree_idx) == 0 else (int((1 - forget_rate) * len(disagree_idx)) or len(disagree_idx))
    disagree_idx = disagree_idx or np.arange(y_hat_1.shape[1])

    y_hat_1_dis, y_hat_2_dis, y_dis, y_hat_lens_dis, target_lens_dis = y_hat_1[:, disagree_idx], y_hat_2[:, disagree_idx], y[disagree_idx], y_hat_lens[disagree_idx], target_lens[disagree_idx]

    losses_1 = F.ctc_loss(y_hat_1_dis.cpu(), y_dis.cpu(), y_hat_lens_dis.cpu(), target_lens_dis.cpu(), zero_infinity=True, reduction='none')
    losses_2 = F.ctc_loss(y_hat_2_dis.cpu(), y_dis.cpu(), y_hat_lens_dis.cpu(), target_lens_dis.cpu(), zero_infinity=True, reduction='none')

    idx_1 = torch.argsort(losses_1)[:num_remember]
    idx_2 = torch.argsort(losses_2)[:num_remember]

    loss_1 = losses_1[idx_2].mean()
    loss_2 = losses_2[idx_1].mean()

    return loss_1 + loss_2


def get_loss(args):
    if args['loss'] == 'ctc':
        return ctc_loss
    elif args['loss'] == 'ctc_custom':
        return CTC()
    elif args['loss'] == 'abstention':
        return AbstentionCTC()
    else:
        return disagreement_loss

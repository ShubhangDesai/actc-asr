import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from models import get_model
from data import get_loader
from losses import *
from optimizers import *
from utils import error_rates

import argparse, json, os, sys
from tensorboardX import SummaryWriter

import numpy as np

def get_parser():
    parser = argparse.ArgumentParser()

    # Experiment Parameters
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--datadir', type=str, required=True)

    # Model Parameters
    parser.add_argument('--num_layers', default=1, type=int)
    parser.add_argument('--num_nodes', default=64, type=int)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--qrnn', action='store_true')
    parser.add_argument('--zoneout', default=0, type=float)
    parser.add_argument('--layer_norm', action='store_true')

    # Data Parameters
    parser.add_argument('--features', type=str, required=True)
    parser.add_argument('--uniform', action='store_true')
    parser.add_argument('--spatial_invariant', action='store_true')
    parser.add_argument('--height', default=127, type=int)
    parser.add_argument('--old_bez', action='store_true')
    parser.add_argument('--delta', action='store_true')

    # Training Parameters
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--opt', default='adam', type=str)

    # Loss Paramters
    parser.add_argument('--loss', default='ctc', type=str)
    parser.add_argument('--meta', action='store_true')

    return parser

def prepare_dir(args):
    if args['name'] == 'test': return

    os.makedirs('exps/' + args['name'])
    os.makedirs('exps/' + args['name'] + '/checkpoints')

    with open('exps/' + args['name'] + '/args.csv', 'w') as f:
        json.dump(args, f)

def train(model, loader, optimizer, loss_fn, forget_rate, e, writer):
    model.train()

    train_loss = 0.0
    num_word, err_word, num_char, err_char = 0, 0, 0, 0
    for i, (X, y, y_hat_lens, target_lens) in enumerate(loader):
        X, y, y_hat_lens, target_lens = Variable(X), y, y_hat_lens, target_lens
        optimizer.zero_grad()

        y_hat = model(X)#, ret = model(X)
        #w, y_hat_lens = ret if 'tuple' in str(type(ret)) else (None, ret)
        loss = loss_fn(y_hat, y, y_hat_lens, target_lens)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 9)
        optimizer.step()

        train_loss += loss.item()

        curr_num_word, curr_err_word, curr_num_char, curr_err_char = error_rates(y_hat, y, target_lens, verbose=True)
        num_word += curr_num_word
        err_word += curr_err_word
        num_char += curr_num_char
        err_char += curr_err_char

        sys.stdout.write('\rEpoch %i: %i/%i' % (e+1, i+1, len(loader)))

        del X, y, y_hat, target_lens, y_hat_lens, loss
        break
        #torch.cuda.empty_cache()

    word_acc = 1.0 - float(err_word) / num_word
    char_acc = 1.0 - float(err_char) / num_char
    train_loss /= len(loader)

    writer.add_scalar('train/char_acc', char_acc, e)
    writer.add_scalar('train/word_acc', word_acc, e)
    writer.add_scalar('train/loss', train_loss, e)

    print('Epoch %i' % (e+1))
    print('Loss\t%.3f' % train_loss)
    print('WAcc\t%.3f' % word_acc)
    print('CAcc\t%.3f' % char_acc)

def eval(model, loader, loss_fn, e, writer):
    model.eval()

    val_loss = 0.0
    num_word, err_word, num_char, err_char = 0, 0, 0, 0
    for i, (X, y, y_hat_lens, target_lens) in enumerate(loader):
        X, y, target_lens = Variable(X), y, target_lens

        y_hat = model(X)#, ret = model(X)
        #y_hat_lens = ret[1] if 'tuple' in str(type(ret)) else ret
        #val_loss += loss_fn(y_hat, y, y_hat_lens, target_lens, 0.0).item()
        val_loss += loss_fn(y_hat, y, y_hat_lens, target_lens)

        curr_num_word, curr_err_word, curr_num_char, curr_err_char = error_rates(y_hat, y, target_lens, verbose=True)
        num_word += curr_num_word
        err_word += curr_err_word
        num_char += curr_num_char
        err_char += curr_err_char

        del X, y, y_hat, target_lens, y_hat_lens
        break
        #torch.cuda.empty_cache()

    word_acc = 1.0 - float(err_word) / num_word
    char_acc = 1.0 - float(err_char) / num_char
    val_loss /= len(loader)

    writer.add_scalar('val/char_acc', char_acc, e)
    writer.add_scalar('val/word_acc', word_acc, e)
    #writer.add_scalar('val/loss', val_loss, e)

    print('Val %i' % (e+1))
    print('Loss\t%.3f' % val_loss)
    print('WAcc\t%.3f' % word_acc)
    print('CAcc\t%.3f' % char_acc)

    return char_acc, word_acc

if __name__ == '__main__':
    args = vars(get_parser().parse_args())
    prepare_dir(args)

    train_loader, val_loader = get_loader(args, 'train'), get_loader(args, 'val')
    model = get_model(args)
    loss_fn = get_loss(args)#ctc_loss

    optimizer = get_opt(args, model)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    writer = SummaryWriter(logdir='tensorboard/' + args['name'])
    best_word_acc = 0.0
    for e in range(args['epochs']):
        if e == 5: loss_fn = get_loss(args)
        forget_rate = min(float(e - 5) / 10 * 0.2, 0.2)

        train(model, train_loader, optimizer, loss_fn, forget_rate, e, writer)
        char_acc, word_acc = eval(model, val_loader, loss_fn, e, writer)

        if word_acc > best_word_acc and args['name'] != 'test':
            best_word_acc = word_acc
            torch.save(model.state_dict(), 'exps/' + args['name'] + '/checkpoints/%i_%i_%i.pth'%(e+1, int(word_acc*100), int(char_acc*100)))

        scheduler.step(word_acc)

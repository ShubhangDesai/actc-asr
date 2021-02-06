import numpy as np
import editdistance, itertools

chars = ['#', '@', ' ', '!', '"', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', ']', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
char_idxs = {c:i for i, c in enumerate(chars)}
#char_idxs = {'#': 0, ' ': 1, '!': 2, '"': 3, '&': 4, "'": 5, '(': 6, ')': 7, '*': 8, '+': 9, ',': 10, '-': 11, '.': 12, '/': 13, '0': 14, '1': 15, '2': 16, '3': 17, '4': 18, '5': 19, '6': 20, '7': 21, '8': 22, '9': 23, ':': 24, ';': 25, '?': 26, 'A': 27, 'B': 28, 'C': 29, 'D': 30, 'E': 31, 'F': 32, 'G': 33, 'H': 34, 'I': 35, 'J': 36, 'K': 37, 'L': 38, 'M': 39, 'N': 40, 'O': 41, 'P': 42, 'Q': 43, 'R': 44, 'S': 45, 'T': 46, 'U': 47, 'V': 48, 'W': 49, 'X': 50, 'Y': 51, 'Z': 52, '[': 53, ']': 54, 'a': 55, 'b': 56, 'c': 57, 'd': 58, 'e': 59, 'f': 60, 'g': 61, 'h': 62, 'i': 63, 'j': 64, 'k': 65, 'l': 66, 'm': 67, 'n': 68, 'o': 69, 'p': 70, 'q': 71, 'r': 72, 's': 73, 't': 74, 'u': 75, 'v': 76, 'w': 77, 'x': 78, 'y': 79, 'z': 80}

def encode_string(string):
    return [char_idxs[c] for c in string]

def get_wer(label_str, pred_str):
    word_idx = {word:chr(i) for i, word in enumerate(set(label_str.split(' ')) | set(pred_str.split(' ')))}
    str_to_idx = lambda s: ''.join(word_idx[word] for word in s.split(' '))

    label_idx, pred_idx = str_to_idx(label_str), str_to_idx(pred_str)
    err_word = editdistance.eval(label_idx, pred_idx)
    num_word = max(len(label_idx), len(pred_idx))

    return num_word, err_word

def decode_indexes(label):
    return ''.join([chars[i] for i in label])

def greedy_decode(pred):
    pred_str = ''.join(chars[i] for i in pred.argmax(1))
    collapsed_str = ''.join(c for c, _ in itertools.groupby(pred_str))
    final_str = collapsed_str.replace('@', '').replace('#', '')

    return final_str

def error_rates(y_hat, y, target_lens, verbose=False):
    num_word, err_word, num_char, err_char = 0, 0, 0, 0

    if len(y_hat) == 2: y_hat = y_hat[0]
    for i in range(y.shape[0]):
        pred = y_hat[:, i].data.cpu().numpy()
        label = y[i].cpu().numpy()
        target_len = target_lens[i]

        pred_str = greedy_decode(pred)
        label_str = decode_indexes(label)[:target_len]

        curr_num_word, curr_err_word = get_wer(label_str, pred_str)
        curr_err_char = editdistance.eval(label_str, pred_str)

        num_word += curr_num_word
        err_word += curr_err_word
        num_char += max(len(label_str), len(pred_str))
        err_char += curr_err_char

        if verbose: print(('' if curr_err_char == 0 else '[ERR] ') + label_str + ' -> ' + pred_str)

    return num_word, err_word, num_char, err_char

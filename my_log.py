import os
import sys
import time
import numpy as np
import torch
import torch.nn.parallel
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import torchvision.transforms as T
import torch.nn.functional as F
import matplotlib.pyplot as plt


# Logging

def gray(x):    return '\033[90m' + str(x) + '\033[0m'
def red(x):     return '\033[91m' + str(x) + '\033[0m'
def green(x):   return '\033[92m' + str(x) + '\033[0m'
def yellow(x):  return '\033[93m' + str(x) + '\033[0m'
def blue(x):    return '\033[94m' + str(x) + '\033[0m'
def magenta(x): return '\033[95m' + str(x) + '\033[0m'
def cyan(x):    return '\033[96m' + str(x) + '\033[0m'
def white(x):   return '\033[97m' + str(x) + '\033[0m'

def fmt(fn_color, *args, **kwargs):
    tmp = ''
    end = '\n'
    for msg in args:
        if isinstance(msg,float):
            msg = '%.5f' % msg
        tmp += '%s ' % (fn_color(msg))
    for k in kwargs.keys():
        if k == 'end':
            end = kwargs['end']
        else:
            msg = kwargs[k]
            if isinstance(msg,float):
                msg = '%.5f' % msg
            tmp += '%s: %s ' % (k, fn_color(msg))
    tmp += end
    return tmp

def print_base(fn_color, *args, **kwargs):
    print(fmt(fn_color, *args, **kwargs),end='')

def debug(*args, **kwargs):
    print_base(gray, *args, **kwargs)

def info(*args, **kwargs):
    print_base(green, *args, **kwargs)

def msg(*args, **kwargs):
    print_base(yellow, *args, **kwargs)

def warn(*args, **kwargs):
    print_base(magenta, *args, **kwargs)

def err(*args, **kwargs):
    print_base(red, *args, **kwargs)


# utils

def mkdir(fn):
    os.makedirs(fn, exist_ok=True)
    return fn

def select_avaliable(fn_list):
    selected = None
    for fn in fn_list:
        if os.path.exists(fn):
            selected = fn
            break
    if selected is None:
        log.err(log.yellow("Could not find dataset from"), fn_list)
    else:
        return selected

# Numpy functions

def num(x):
    return x.detach().cpu().numpy()

def norm_01(x):
    return (x - x.min())/(x.max() - x.min() + + 1e-6)

def relu(x):
    return np.maximum(0, x)

def np_l2_sum(x):
    return np.sqrt(np.square(x.copy()).sum())

def np_l2_mean(x):
    return np.sqrt(np.square(x.copy()).mean())

def np_inf_norm(x):
    return np.linalg.norm(x, ord=np.inf_norm)

def np_clip_by_l2norm(x, clip_norm):
    return x * clip_norm / np.linalg.norm(x, ord=2)

def np_clip_by_infnorm(x, clip_norm):
    return x * clip_norm / np.linalg.norm(x, ord=np.inf)

# Display functions

def print_mat(x):
    info(x.shape, x.dtype, min=x.min(), max=x.max())

def print_l2(x):
    info(x.shape,min=x.min(),max=x.max(),sum_l2 =np_l2_sum(x),mean_l2=np_l2_mean(x))

def get_fig(figsize=(8,4)):
    fig = plt.figure(figsize=figsize, dpi=100, facecolor='w', edgecolor='k')
    return fig
    
def sub_plot(fig, rows, cols, index, title, image):
    axis = fig.add_subplot(rows, cols, index)
    if title != None:
        axis.title.set_text(title)
    axis.axis('off')
    plt.imshow(image)


# Timing

class Tick():
    def __init__(self, name='', silent=False):
        self.name = name
        self.silent = silent

    def __enter__(self):
        self.t_start = time.time()
        if not self.silent:
            print(cyan('> %s ... ' % (self.name)), end='')
            sys.stdout.flush()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.t_end = time.time()
        self.delta = self.t_end-self.t_start
        self.fps = 1/self.delta

        if not self.silent:
            print(cyan('[%.0f ms]' % (self.delta * 1000)))
            sys.stdout.flush()


class Tock():
    def __init__(self, name=None, report_time=True):
        self.name = '' if name == None else name+': '
        self.report_time = report_time

    def __enter__(self):
        self.t_start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.t_end = time.time()
        self.delta = self.t_end-self.t_start
        self.fps = 1/self.delta
        if self.report_time:
            print(yellow('(%s%.0fms) ' % (self.name, self.delta * 1000)), end='')
        else:
            print(yellow('.'), end='')
        sys.stdout.flush()

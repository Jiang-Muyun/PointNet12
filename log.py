import time
import sys

def gray(x):    return '\033[90m' + str(x) + '\033[0m'
def red(x):     return '\033[91m' + str(x) + '\033[0m'
def green(x):   return '\033[92m' + str(x) + '\033[0m'
def yellow(x):  return '\033[93m' + str(x) + '\033[0m'
def blue(x):    return '\033[94m' + str(x) + '\033[0m'
def magenta(x): return '\033[95m' + str(x) + '\033[0m'
def cyan(x):    return '\033[96m' + str(x) + '\033[0m'
def white(x):   return '\033[97m' + str(x) + '\033[0m'

def print_base(fn_color,*args, **kwargs):
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
            tmp += '%s: %s ' % (white(k), fn_color(msg))
    print(tmp)

def debug(*args, **kwargs):
    print_base(gray, *args, **kwargs)

def info(*args, **kwargs):
    print_base(blue, *args, **kwargs)

def warn(*args, **kwargs):
    print_base(magenta, *args, **kwargs)

def err(*args, **kwargs):
    print_base(red, *args, **kwargs)

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


if __name__ == '__main__':
    debug('debug',arg='testing')
    info('info',arg='testing')
    warn('warn',arg='testing')
    err('err',arg='testing')

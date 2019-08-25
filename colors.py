
def grey(x): return '\033[90m' + str(x) + '\033[0m'
def red(x): return '\033[91m' + str(x) + '\033[0m'
def green(x): return '\033[92m' + str(x) + '\033[0m'
def yellow(x): return '\033[93m' + str(x) + '\033[0m'
def blue(x): return '\033[94m' + str(x) + '\033[0m'


def print_kv(*kv):
    for i in range(int(len(kv)/2)):
        print(yellow(kv[i*2]), blue(kv[i*2+1]), end=' ')
    print()

def print_info(*kv, end='\n'):
    tmp = ''
    for msg in kv:
        tmp += green(msg) + ' '
    print(tmp,end=end)

def print_debug(*kv, end='\n'):
    tmp = ''
    for msg in kv:
        tmp += grey(msg) + ' '
    print(tmp,end=end)

def print_err(*kv, end='\n'):
    tmp = ''
    for msg in kv:
        tmp += red(msg) + ' '
    print(tmp,end=end)
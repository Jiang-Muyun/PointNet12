
def grey(x): return '\033[90m' + str(x) + '\033[0m'
def red(x): return '\033[91m' + str(x) + '\033[0m'
def green(x): return '\033[92m' + str(x) + '\033[0m'
def yellow(x): return '\033[93m' + str(x) + '\033[0m'
def blue(x): return '\033[94m' + str(x) + '\033[0m'


def print_kv(key, value, end='\n'):
    print(yellow(key), blue(value), end=end)


def print_info(msg, end='\n'):
    print(green(msg), end=end)


def print_debug(msg, end='\n'):
    print(grey(msg), end=end)


def print_err(msg, end='\n'):
    print(red(msg), end=end)


grey = lambda x: '\033[90m' + str(x) + '\033[0m'
red = lambda x: '\033[91m' + str(x) + '\033[0m'
green = lambda x: '\033[92m' + str(x) + '\033[0m'
yellow = lambda x: '\033[93m' + str(x) + '\033[0m'
blue = lambda x: '\033[94m' + str(x) + '\033[0m'

def print_kv(key,value):
    print(yellow(key),blue(value))

def print_info(msg):
    print(green(msg))

def print_debug(msg):
    print(grey(msg))

def print_err(msg):
    print(red(msg))
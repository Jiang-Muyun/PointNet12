
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
    for msg in args:
        tmp += '%s ' % (fn_color(msg))
    for k in kwargs.keys():
        tmp += '%s: %s ' % (yellow(k), fn_color(kwargs[k]))
    print(tmp)

def debug(*args, **kwargs):
    print_base(blue, *args, **kwargs)

def info(*args, **kwargs):
    print_base(green, *args, **kwargs)

def warn(*args, **kwargs):
    print_base(magenta, *args, **kwargs)

def err(*args, **kwargs):
    print_base(red, *args, **kwargs)

if __name__ == '__main__':
    debug('debug',arg='testing')
    log('log',arg='testing')
    info('info',arg='testing')
    warn('warn',arg='testing')
    err('err',arg='testing')

def LinuxConsoleColor():
    print("\033[0;37;40m Normal text\n")
    print("\033[2;37;40m Underlined text\033[0;37;40m \n")
    print("\033[1;37;40m Bright Colour\033[0;37;40m \n")
    print("\033[3;37;40m Negative Colour\033[0;37;40m \n")
    print("\033[5;37;40m Negative Colour\033[0;37;40m\n")
    
    print("\033[1;37;40m \033[2;37:40m TextColour BlackBackground          TextColour GreyBackground                WhiteText ColouredBackground\033[0;37;40m\n")
    print("\033[1;30;40m Dark Gray      \033[0m 1;30;40m            \033[0;30;47m Black      \033[0m 0;30;47m               \033[0;37;41m Black      \033[0m 0;37;41m")
    print("\033[1;31;40m Bright Red     \033[0m 1;31;40m            \033[0;31;47m Red        \033[0m 0;31;47m               \033[0;37;42m Black      \033[0m 0;37;42m")
    print("\033[1;32;40m Bright Green   \033[0m 1;32;40m            \033[0;32;47m Green      \033[0m 0;32;47m               \033[0;37;43m Black      \033[0m 0;37;43m")
    print("\033[1;33;40m Yellow         \033[0m 1;33;40m            \033[0;33;47m Brown      \033[0m 0;33;47m               \033[0;37;44m Black      \033[0m 0;37;44m")
    print("\033[1;34;40m Bright Blue    \033[0m 1;34;40m            \033[0;34;47m Blue       \033[0m 0;34;47m               \033[0;37;45m Black      \033[0m 0;37;45m")
    print("\033[1;35;40m Bright Magenta \033[0m 1;35;40m            \033[0;35;47m Magenta    \033[0m 0;35;47m               \033[0;37;46m Black      \033[0m 0;37;46m")
    print("\033[1;36;40m Bright Cyan    \033[0m 1;36;40m            \033[0;36;47m Cyan       \033[0m 0;36;47m               \033[0;37;47m Black      \033[0m 0;37;47m")
    print("\033[1;37;40m White          \033[0m 1;37;40m            \033[0;37;40m Light Grey \033[0m 0;37;40m               \033[0;37;48m Black      \033[0m 0;37;48m")
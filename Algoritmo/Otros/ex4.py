import subprocess
import time
import argparse

parser = argparse.ArgumentParser(description='Display WLAN signal strength.')
parser.add_argument(dest='interface', nargs='?', default='wlp2s0',
                    help='wlan interface (default: wlp2s0)')
args = parser.parse_args()

from termcolor import colored

print(colored('\n     P', 'red'), colored('e', 'yellow'), colored('n', 'green'), colored('t', 'white'), colored('e', 'cyan'), colored('s', 'blue'), colored('t', 'magenta'), colored(' - ', 'white'), colored('R', 'red'), colored('a', 'yellow'), colored('b', 'green'), colored('b', 'white'), colored('i', 'cyan'), colored('t\n', 'blue')
)
while True:
    cmd = subprocess.Popen('iwconfig %s' % args.interface, shell=True,
                           stdout=subprocess.PIPE)
    for line in cmd.stdout:
        if 'Link Quality' in line:
            print(line.lstrip(' ')),
        elif 'Not-Associated' in line:
            print('No signal')
    time.sleep(1)
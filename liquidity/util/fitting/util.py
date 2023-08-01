import os
import sys


def block_print():
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    return original_stdout


def enable_print(original_stdout):
    if original_stdout is not None:
        sys.stdout = original_stdout

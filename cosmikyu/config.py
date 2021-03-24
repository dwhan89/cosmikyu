import argparse
import os

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

# default directory list
default_log_dir = os.path.join(_root, "log")
default_data_dir = os.path.join(_root, "data")
default_output_dir = os.path.join(_root, "output")

argparser = argparse.ArgumentParser(
    prog='cosmikyu',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    conflict_handler='resolve'
)

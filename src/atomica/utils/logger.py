#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import sys
import datetime


LEVELS = ['TRACE', 'DEBUG', 'INFO', 'WARN', 'ERROR']
LEVELS_MAP = None


def init_map():
    global LEVELS_MAP, LEVELS
    LEVELS_MAP = {}
    for idx, level in enumerate(LEVELS):
        LEVELS_MAP[level] = idx


def get_prio(level):
    global LEVELS_MAP
    if LEVELS_MAP is None:
        init_map()
    return LEVELS_MAP[level.upper()]


def print_log(s, level='INFO', end='\n', no_prefix=False):
    pth_prio = get_prio(os.getenv('LOG', 'INFO'))
    prio = get_prio(level)
    if prio >= pth_prio:
        if not no_prefix:
            now = datetime.datetime.now()
            prefix = now.strftime("%Y-%m-%d %H:%M:%S") + f'::{level.upper()}::'
            print(prefix, end='')
        print(s, end=end)
        sys.stdout.flush()

#!/usr/bin/python
# -*- coding:utf-8 -*-
from time import time


def get_time_sign(prefix='', suffix='') -> str:
    time_note = time()
    unique_id = round(round(time_note - round(time_note), 3) * 1000)
    unique_id = unique_id if unique_id > 0 else -unique_id
    return prefix + str(unique_id) + suffix
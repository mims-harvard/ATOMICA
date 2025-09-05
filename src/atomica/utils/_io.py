#!/usr/bin/python
# -*- coding:utf-8 -*-
from os.path import basename, splitext
import pandas as pd


def filename(path):
    return basename(splitext(path)[0])


def read_csv(fpath, sep=','):
    heads, entries = [], []
    df = pd.read_csv(fpath, sep=sep)
    heads = list(df.columns)
    for rid in range(len(df)):
        entry = []
        for h in heads:
            entry.append(str(df[h][rid]))
        entries.append(entry)
    return heads, entries
#!/usr/bin/python
# -*- coding:utf-8 -*-
'''
R = 8.31446261815324 J.mol-1.k-1
1 kcal = 4184 J
=> R = 0.0019872043 kcal.mol-1.-1
'''
import math

R = 0.0019872043

def dg_to_kd(dg, temperature=25.0):
    """Coversion of DG into the dissociation constant kd """
    
    temp_in_k = temperature + 273.15
    rt = R * temp_in_k
    return math.exp(dg / rt)


def kd_to_dg(kd, temperature=25.0):
    """Conversion of Kd to DG"""
    dg_rt = math.log(kd)
    temp_in_k = temperature + 273.15
    rt = R * temp_in_k
    return dg_rt * rt


def cal_temperature(dg, kd):
    dg_rt = math.log(kd)
    rt = dg / dg_rt
    temp_in_k = rt / R
    temperature = temp_in_k - 273.15
    return temperature

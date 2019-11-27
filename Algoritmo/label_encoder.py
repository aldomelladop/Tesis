#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 17:28:17 2019

@author: aldo_mellado
"""
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

le = preprocessing.LabelEncoder()
a = le.fit(["paris", "paris", "tokyo", "amsterdam"])

list(le.classes_)

le.transform(["tokyo", "tokyo", "paris"]) 

list(le.inverse_transform([2, 2, 1]))
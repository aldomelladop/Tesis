#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 14:54:48 2019

@author: aldo_mellado
"""

scifi_authors = ["Isaac Asimov", "Ray Bradbury", "Robert Heinlein",
                 "Arts C. Clarke", "Frank Herbert", "Orson Scott Card", 
                 "Douglas Adams", "H. G. Wells", "Leigh Brackett"]

scifi_authors.sort(key=lambda name: name.split(" ")[-1].lower())


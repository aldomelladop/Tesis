#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 19:23:25 2019

@author: aldo_mellado
"""

import logging

# Create and configure logger

logging.basicConfig(filename = "/home/aldo_mellado/Documents/2019-1/Tesis/Algoritmo/error.log",
                    format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)
logging.info('Admin logged in')
# =============================================================================
#  NOTSET =  0
#  DEBUG = 10
#  INFO = 20
#  WARNING  = 30
#  ERROR = 40
#  CRITICAL = 50
# =============================================================================


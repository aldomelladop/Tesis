#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 09:41:27 2019

@author: aldo_mellado
"""

fibonacci_cache = {}

def fibonacci(n):
    # If we have cached the value, then return it
    if n in fibonacci_cache:
        return fibonacci_cache[n]

    #Compute the Nth term
    if n==1:
        value = 1
    elif n==2:
        value = 2
    elif n>2:
        value = fibonacci(n-1) + fibonacci(n-2)
    
    #Cache the value an return it
    fibonacci_cache[n] = value
    return value

for n in range (1,105):
    print(n,": ",fibonacci(n))
    
    
# =============================================================================
# LRU_cache
# =============================================================================
    
from functools import lru_cache

@lru_cache(maxsize = 1000)

def fibonacci(n):
    #Check that the imput is a positive integer
    if type(n) != int:
        raise TypeError("n must be a positive int")
    if n < 1:
        raise ValueError("n must be a postive int")
    
    #Compute the Nth term
    if n==1:
        return 1
    elif n==2:
        return 1
    elif n>2:
        return fibonacci(n-1) + fibonacci(n-2)
    
print(fibonacci(1001)/fibonacci(1000))
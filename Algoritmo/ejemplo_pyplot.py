#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 00:16:13 2019

@author: aldo_mellado
"""
import matplotlib.pyplot as plt

t = np.arange(0.,5.,0.5)

plt.subplot(131)
plt.plot(t,t, 'ro')
plt.subplot(132)
plt.plot(t,t**2,'bs')
plt.subplot(133)
plt.plot(t,t**3,'g^')
plt.suptitle('some numbers')
plt.show()

# =============================================================================
# 
# =============================================================================
def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)

t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)

plt.figure()
plt.subplot(211)
plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')

plt.subplot(212)
plt.plot(t2, np.cos(2*np.pi*t2), 'r--')
plt.show()
Copy to clipboard

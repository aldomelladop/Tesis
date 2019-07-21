#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 13:17:33 2019

@author: aldo_mellado
"""

from getkey import getkey, keys
key = getkey()
while(1):
    if key == keys.UP:
        for i in range(0,10):
            print(i)
    elif key == keys.DOWN:
        for i in range(10,20):
            print(i)
    else:
        print("Continue...")
    
    print("Press 'q' to exit\n")
    key = getkey()
    if key=='q':
        break
        
    
    	print("Press 'c' to continue or 'q' to quit")
    	flag = False
    	key = getkey()

    	while(flag==False):
    		if key == 'c':
    			flag=True
    			continue
	    	elif key == 'q':
	        	print("Quit\n")
                flag=
	        	break
	    	else:
	        	print("Please, choose a valid option")
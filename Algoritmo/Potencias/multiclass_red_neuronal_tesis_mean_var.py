#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 17:14:06 2019

@author: aldo_mellado
"""

# =============================================================================
# Importing the libraries
# =============================================================================
import os
import numpy as np
import pandas as pd
# =============================================================================
# from fixrows import fixrows
# from merge_csv import fusionar_csv
# from create_cords import createcords
# =============================================================================

# =============================================================================
# Importing the dataset
# =============================================================================
   
#Run this code, only if the file with all the dataframes were deleted

# =============================================================================
# dir_pot = os.getcwd() + '/Potencias/'
#  
df1 = fixrows('Potencia_r1').iloc[:150,:]
num_row = np.shape(df1)[0]
coords  = ['(1,0)' for j in range(num_row)]
coords = pd.DataFrame(coords,dtype=object, columns = ['X,Y'])
df1 = coords.join(df1, how='left')
df1.to_csv(dir_pot + 'Potencia_R1.csv')
# 
# df2 = fixrows(dir_pot+'Potencia_r2')
# num_row = np.shape(df2)[0]
# coords  = ['(0,0)' for j in range(num_row)]
# coords = pd.DataFrame(coords,dtype=object, columns = ['X,Y'])
# df2 = coords.join(df2, how='left')
# df2.to_csv(dir_pot + 'Potencia_R2.csv')
# 
# df3 = fixrows(dir_pot+'Potencia_r3')
# num_row = np.shape(df3)[0]
# coords  = ['(0,1)' for j in range(num_row)]
# coords = pd.DataFrame(coords,dtype=object, columns = ['X,Y'])
# df3 = coords.join(df3, how='left')
# df3.to_csv(dir_pot + 'Potencia_R3.csv')
# 
# df4 = fixrows(dir_pot+'Potencia_r4')
# num_row = np.shape(df4)[0]
# coords  = ['(1,1)' for j in range(num_row)]
# coords = pd.DataFrame(coords,dtype=object, columns = ['X,Y'])
# df4 = coords.join(df4, how='left')
# df4.to_csv(dir_pot + 'Potencia_R4.csv')
#  
#  #Ingresar a carpeta potencias
# fusionar_csv('Potencia_R1','Potencia_R2','Potencia_R3','Potencia_R4')
# 
# df0 = fixrows(dir_pot+'/potencias_fusionado').iloc[:,1:]

df0 = pd.read_csv('potencias_fusionado_corregido.csv').iloc[:,1:]

X = df0.iloc[:,1:].values #variables Dependientes (Potencias)
y = df0.iloc[:,0].values #values Independientes (Posición)

from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
encoder.fit(y)
y_encoded = encoder.transform(y)
y_encoded = np_utils.to_categorical(y_encoded)

#returns a list with the positions (0,0),(0,1),(1,0),(1,1)
#list(encoder.classes_)
#encoder.transform(['(0,0)','(0,0)','(1,0)','(1,1)']) 
#list(encoder.inverse_transform([2, 2, 1]))

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size = 0.3, random_state = 0)

# =============================================================================
# Feature Scaling (Standarization)
# =============================================================================
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_trainn = sc.fit_transform(X_train)
X_testn = sc.transform(X_test)
# =============================================================================


# Feature Scaling (Normalization)
#from sklearn import preprocessing
#X_trainn = preprocessing.normalize(X_train)
#X_testn = preprocessing.normalize(X_test)
# =============================================================================


import keras
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu', input_dim = 19))
    classifier.add(Dropout(rate = 0.35))
    
    classifier.add(Dense(units = 18, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(rate = 0.35))
    
    classifier.add(Dense(units = 12, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(rate = 0.35))
    
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))   
    classifier.add(Dropout(rate = 0.35))

    classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'softmax'))
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 16, epochs = 5)
accuracies = cross_val_score(estimator = classifier, X = X_trainn, y = y_train, cv = 10, n_jobs = -1)
mean = accuracies.mean()
variance = accuracies.std()

history = classifier.fit(X_trainn, y_train, batch_size = 16, epochs = 5, validation_split=0.2)


# =============================================================================
#                               Confusion matrix
# =============================================================================
# =============================================================================
# from sklearn.metrics import multilabel_confusion_matrix
# 
# y_pred = classifier.predict(y_test,y_pred)
# =============================================================================


from matplotlib import pyplot as plt

ax1 = plt.subplot(121)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='center right')
plt.grid()
ax1.set_ylim([0.9, 1.02])

plt.subplot(122)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='center right')
plt.grid()
plt.subplots_adjust(wspace =0.4, hspace= 2.5)
plt.savefig('accuracy_over_epochs_train.pdf')

# =============================================================================
# y_pred = classifier.predict(X_test)
# y_pred = (y_pred > 0.5)
# 
# new_prediction = classifier.predict(np.array([[-67.0, -69, -73, -75, -81, -61, -71, -65, -75, -85, -73, -63, -53,-71, -71, -81, -60]]))

prediction_ = np.argmax(np_utils.to_categorical(X_testn[:1,0:]), axis = 1)
predictions = list(encoder.inverse_transform(prediction_))

X_test_list = X_testn[:1,0:]
for i in X_testn[0,:]:
    print(sc.inverse_transform(i))


# =============================================================================
    
#from ann_visualizer.visualize import ann_viz

#ann_viz(classifier, view=True, filename='network.gv', title='MyNeural Network_mean_var')
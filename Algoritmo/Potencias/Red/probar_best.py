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
import random
import numpy as np
import pandas as pd
from pytictoc import TicToc
from fixrows  import fixrows
from merge_csv import fusionar_csv

# =============================================================================
# Importing the dataset
# =============================================================================
   
#Run this code, only if the file with all the dataframes were deleted


t.tic()
print(f"Probando {j}\n")

directory = os.getcwd()

if os.path.isdir(os.getcwd() + '/' + str(j)) == True:
	pass
else:
	os.mkdir(os.getcwd() + '/' + str(j))

df1 = fixrows('Potencia_r1').iloc[:j,:]
num_row = np.shape(df1)[0]
coords  = ['(0,0)' for j in range(num_row)]
coords = pd.DataFrame(coords,dtype=object, columns = ['X,Y'])
df1 = coords.join(df1, how='left')
df1.to_csv('Potencia_R1.csv')

df2 = fixrows('Potencia_r2').iloc[:j,:]
num_row = np.shape(df2)[0]
coords  = ['(0,1)' for j in range(num_row)]
coords = pd.DataFrame(coords,dtype=object, columns = ['X,Y'])
df2 = coords.join(df2, how='left')
df2.to_csv('Potencia_R2.csv')

df3 = fixrows('Potencia_r3').iloc[:j,:]
num_row = np.shape(df3)[0]
coords  = ['(1,0)' for j in range(num_row)]
coords = pd.DataFrame(coords,dtype=object, columns = ['X,Y'])
df3 = coords.join(df3, how='left')
df3.to_csv('Potencia_R3.csv')

df4 = fixrows('Potencia_r4').iloc[:j,:]
num_row = np.shape(df4)[0]
coords  = ['(1,1)' for j in range(num_row)]
coords = pd.DataFrame(coords,dtype=object, columns = ['X,Y'])
df4 = coords.join(df4, how='left')
df4.to_csv('Potencia_R4.csv')

#Ingresar a carpeta potencias
t.tic()
fusionar_csv('Potencia_R1','Potencia_R2','Potencia_R3','Potencia_R4')

t.toc('\nTiempo Archivos Fusionados\n')

df0 = fixrows('potencias_fusionado').iloc[3:,1:]

#    df0 = pd.read_csv('potencias_fusionado_corregido.csv').iloc[3:,1:]

X = df0.iloc[:,1:].values #variables Dependientes (Potencias)
y = df0.iloc[:,0].values #values Independientes (Posición)

import keras
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
encoder.fit(y)
y_encoded = encoder.transform(y)
y_encoded = np_utils.to_categorical(y_encoded)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size = 0.2, random_state = 0)

# =============================================================================
# Feature Scaling (Standarization)
# =============================================================================
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense

# =============================================================================
#  insertar mejores parámetros
# =============================================================================

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense

best_parameters = {}

# =============================================================================
# Cross Validation
# =============================================================================

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

t.tic() 
print("\nEntrando en Cross Validation\n")

best_parameters = {'batch_size': 32, 'epochs': 30, 'optimizer': 'adam'} #1000
#best_parameters = {'batch_size': 32, 'epochs': 25, 'optimizer': 'adam'} #5000
#best_parameters = {'batch_size': 16, 'epochs': 15, 'optimizer': 'adam'} #10000
#best_parameters = {'batch_size': 32, 'epochs': 30, 'optimizer': 'adam'} #15000
j = 1000

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = int(np.shape(X_test)[1]/2)+1, kernel_initializer = 'uniform', activation = 'relu', input_dim = np.shape(X_test)[1]))
    classifier.add(Dropout(rate = 0.3))
    
    classifier.add(Dense(units = 18, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(rate = 0.3))
    
    classifier.add(Dense(units = 24, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(rate = 0.3))
    
    classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu'))    
    classifier.add(Dropout(rate = 0.3))

    classifier.add(Dense(units = 12, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(rate = 0.3))

    classifier.add(Dense(units = int(np.shape(y_test)[1]), kernel_initializer = 'uniform', activation = 'softmax'))
    classifier.compile(optimizer = best_parameters['optimizer'], loss = 'categorical_crossentropy', metrics = ['accuracy'])
    # classifier.compile(optimizer = 'adamax', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = best_parameters['batch_size'], epochs = best_parameters['epochs'])
    
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
ac = list(accuracies)
mean = accuracies.mean()
variance = accuracies.std()
t.toc('\nTiempo de red neuronal: ')
time = t.tocvalue()

# =============================================================================
#     Saving model
# =============================================================================
classifier = Sequential()
classifier.add(Dense(units = int(np.shape(X_test)[1]/2)+1, kernel_initializer = 'uniform', activation = 'relu', input_dim = np.shape(X_test)[1]))
classifier.add(Dropout(rate = 0.3))

classifier.add(Dense(units = 18, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(rate = 0.3))

classifier.add(Dense(units = 24, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(rate = 0.3))

classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu'))    
classifier.add(Dropout(rate = 0.3))

classifier.add(Dense(units = 12, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(rate = 0.3))

classifier.add(Dense(units = int(np.shape(y_test)[1]), kernel_initializer = 'uniform', activation = 'softmax'))
classifier.compile(optimizer = best_parameters['optimizer'], loss = 'categorical_crossentropy', metrics = ['accuracy'])

from keras.models import load_model

# save model and architecture to single file
classifier.save(directory + "/{}/model_{}.h5".format(j,j))

print("Saved model to disk\n")

# =============================================================================
#     Escritura de archivo
# =============================================================================
print(f"\nnp.shape(X_test)[0] = {np.shape(X_test)[0]}\n")

outFileName= directory + "/{}/resultados_{}.txt".format(j,j)
f = open(outFileName,"w")

f.write("El número de elementos usados es: {}\n".format(j)+
        "Los mejores parámetros son: {}".format(best_parameters) +
        "\nTiempo de GridSearchCV  = {}".format(round(int(t1/60),2)) + 
        "\nTiempo red neuronal  =  {}".format(time) + 
        "\nLa media obtenida es: {}".format(mean)+ 
        "\nLa varianza obtenida es: {}".format(variance) + '\n'
        )
for i in ac:
    f.write("\tac: " +  repr(round((i*100),2)) +"%" + '\n')

# =============================================================================
#                             Prediction
# =============================================================================
for i in range(1,15):
    r = random.randint(0, np.shape(X_test)[0]-1)
    y_pred = classifier.predict(np.array([X_test[r]]))
    predictions = list(encoder.inverse_transform([np.argmax(y_pred, axis=None, out=None)]))
    y_pred_prob = classifier.predict_proba(np.array([X_test[r]]))
    f.write("\nFor the vector: ["+ repr(X_test[r])+ "]\t the predicted position is:" +  repr(predictions) +  "and its accuracy was:" + repr(round(np.amax(y_pred_prob),2)))
f.close()

print("Archivo escrito\n")

# =============================================================================
# Confussion Matri6x
# =============================================================================
    
# y_pred = list(encoder.inverse_transform([np.argmax(classifier.predict(np.array(X_test)))]))
# y_real  = encoder.inverse_transform([np.argmax(i, axis=None, out=None) for i in y_test])

# =============================================================================
# Funcion de Prueba
# =============================================================================

from plot_history import * 
from full_multiclass_report import * 

history = classifier.fit(X_train, 
                    y_train,
                    epochs = best_parameters['epochs'],
                    batch_size = best_parameters['batch_size'],
                    verbose=0,
                    validation_data=(X_test,y_test))
plot_history(history)

full_multiclass_report(classifier,X_test,y_test,classes=['(0,0)','(0,1)','(1,0)','(1,1)'])


# =============================================================================
#     Move file to folder
# =============================================================================
mv = directory + '/{}/Confusion_matrix.png'.format(j)
os.system('mv Confusion_matrix.png '+ mv)

mv = directory + '/{}/Classification_report.csv'.format(j)
os.system('mv Classification_report.csv ' + mv)

# =============================================================================
#< Plot loss and accuracy
# =============================================================================
from matplotlib import pyplot as plt

history = classifier.fit(X_train, y_train, validation_split=0.2, epochs=best_parameters['epochs'], batch_size=best_parameters['batch_size'], verbose=0)

# list all data in history
print(history.history.keys())

# summarize history for accuracy
ax1 = plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()
#plt.close()

# summarize history for loss
ax2 = plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()
#plt.close()
        
plt.savefig( os.getcwd() + '/{}/accuracy_over_epochs_train_{}.pdf'.format(j,j))

# =============================================================================
#   Load Model
# =============================================================================
# load and evaluate a saved model
# from numpy import loadtxt
# from keras.models import load_model

# # load model
# classifier = load_model('model.h5')
# # summarize model.
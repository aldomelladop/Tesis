#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 17:14:06 2019

@author: aldo_mellado
"""
# ============================================================================
# Importing the libraries
# =============================================================================
import os
import random
import numpy as np
import pandas as pd
from pytictoc import TicToc
from fixrows import fixrows
from merge_csv import fusionar_csv
from keras.models import model_from_json
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
# =============================================================================
# Importing the dataset
# =============================================================================
j = int(input("Ingrese j: "))

directory = os.getcwd()

if os.path.isdir(os.getcwd() + '/{}'.format(j)) == True:
    pass
else:
    os.mkdir(os.getcwd() + '/{}'.format(j))

t = TicToc()
df1 = fixrows('Potencia_r1').iloc[:j,:]
num_row = np.shape(df1)[0]
coords  = ['(1,0)' for j in range(num_row)]
coords = pd.DataFrame(coords,dtype=object, columns = ['X,Y'])
df1 = coords.join(df1, how='left')
df1.to_csv('Potencia_R1.csv')
 
df2 = fixrows('Potencia_r2').iloc[:j,:]
num_row = np.shape(df2)[0]
coords  = ['(0,0)' for j in range(num_row)]
coords = pd.DataFrame(coords,dtype=object, columns = ['X,Y'])
df2 = coords.join(df2, how='left')
df2.to_csv('Potencia_R2.csv')
 
df3 = fixrows('Potencia_r3').iloc[:j,:]
num_row = np.shape(df3)[0]
coords  = ['(0,1)' for j in range(num_row)]
coords = pd.DataFrame(coords,dtype=object, columns = ['X,Y'])
df3 = coords.join(df3, how='left')
df3.to_csv('Potencia_R3.csv')

df4 = fixrows('Potencia_r4').iloc[:j,:]
num_row = np.shape(df4)[0]
coords  = ['(1,1)' for j in range(num_row)]
coords = pd.DataFrame(coords,dtype=object, columns = ['X,Y'])
df4 = coords.join(df4, how='left')
df4.to_csv('Potencia_R4.csv')
  
# Fusionar archivos corregidos para obtener el archivo de potencias final
fusionar_csv('Potencia_R1','Potencia_R2','Potencia_R3','Potencia_R4')
 
df0 = fixrows('potencias_fusionado').iloc[:,1:]
os.system('rm Potencia_R*')
os.system('rm Potencia_r*_corregido.csv')

df0 = pd.read_csv('potencias_fusionado_corregido.csv').iloc[:,1:]

X = df0.iloc[:,1:].values #variables Dependientes (Potencias)
y = df0.iloc[:,0].values #values Independientes (Posición)

from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
encoder.fit(y)
y_encoded = encoder.transform(y)
y_encoded = np_utils.to_categorical(y_encoded)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size = 0.2)

# =============================================================================
# Feature Scaling (Standarization)
# =============================================================================
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

# Feature Scaling (Normalization)
from sklearn import preprocessing
X_train = preprocessing.normalize(X_train)
X_test  = preprocessing.normalize(X_test)

# =============================================================================
#  Buscar mejores parámetros
# =============================================================================

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense

# print("Comenzando Grid_search\n")
# t.tic()

# def build_classifier(optimizer):
#     classifier = Sequential()
# #1
#     classifier.add(Dense(units = int(np.shape(X_test)[1])+1, kernel_initializer = 'uniform', activation = 'relu', input_dim = np.shape(X_test)[1]))
#     classifier.add(Dropout(rate = 0.3))
# #2
#     classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu'))
#     classifier.add(Dropout(rate = 0.3))
# #3
#     classifier.add(Dense(units = 24, kernel_initializer = 'uniform', activation = 'relu'))
#     classifier.add(Dropout(rate = 0.3))
# #4
#     classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu'))    
#     classifier.add(Dropout(rate = 0.3))
# #5
#     classifier.add(Dense(units = 12, kernel_initializer = 'uniform', activation = 'relu'))
#     classifier.add(Dropout(rate = 0.3))
# #7
#     classifier.add(Dense(units = np.shape(y_test)[1], kernel_initializer = 'uniform', activation = 'softmax'))
#     classifier.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
#     return classifier

# classifier = KerasClassifier(build_fn = build_classifier)
# parameters = {'batch_size': [16,32, 64],
#               'epochs': [10, 15, 20],
#                 'optimizer': ['adam', 'adamax','rmsprop']}

# grid_search = GridSearchCV(estimator = classifier,
#                             param_grid = parameters,
# #                           scoring = 'accuracy',
#                             cv = 15,
#                             n_jobs = -1)

# grid_search = grid_search.fit(X_train, y_train)

# best_parameters  = grid_search.best_params_
# print(f"best_parameters = {grid_search.best_params_}")
# print(f"best_accuracy =   {grid_search.best_score_}")
# t.toc('Finalizado - Grid_search')
# t1 = t.tocvalue()

# =============================================================================
# Cross Validation
# =============================================================================

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

best_parameters = {'batch_size': 32, 'epochs': 30, 'optimizer': 'adam'}

t.tic() 
print("\nEntrando en Cross Validation\n")

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = int(np.shape(X_test)[1])+1, kernel_initializer = 'uniform', activation = 'relu', input_dim = np.shape(X_test)[1]))
    classifier.add(Dropout(rate = 0.3))
    
    classifier.add(Dense(units = 18, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(rate = 0.3))
    
    classifier.add(Dense(units = 24, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(rate = 0.3))
    
    classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu'))    
    classifier.add(Dropout(rate = 0.3))

    classifier.add(Dense(units = 12, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(rate = 0.3))

    classifier.add(Dense(units = np.shape(y_test)[1], kernel_initializer = 'uniform', activation = 'softmax'))
    classifier.compile(optimizer = best_parameters['optimizer'], loss = 'categorical_crossentropy', metrics = ['accuracy'])
#    classifier.compile(optimizer = 'adamax', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = best_parameters['batch_size'], epochs = best_parameters['epochs'])

accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
ac = list(accuracies)
mean = accuracies.mean()
variance = accuracies.std()
t.toc('\tCross Validation:  ')
time = t.tocvalue()

# =============================================================================
#     Saving model
# =============================================================================
classifier = Sequential()
classifier.add(Dense(units = int(np.shape(X_test)[1])+1, kernel_initializer = 'uniform', activation = 'relu', input_dim = np.shape(X_test)[1]))
classifier.add(Dropout(rate = 0.3))

classifier.add(Dense(units = 18, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(rate = 0.3))

classifier.add(Dense(units = 24, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(rate = 0.3))

classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu'))    
classifier.add(Dropout(rate = 0.3))

classifier.add(Dense(units = 12, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(rate = 0.3))

classifier.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'softmax'))
classifier.compile(optimizer = best_parameters['optimizer'], loss = 'categorical_crossentropy', metrics = ['accuracy'])

from keras.models import model_from_json
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
    
# y_pred
    #Transforma mediciones estandarizadas de potencia, en valores codificados
    # de y, que finalmente son transformados en potencias.
    
# y_pred = list(encoder.inverse_transform(classifier.predict(np.array(X_test))))
y_pred = list(encoder.inverse_transform([np.argmax(classifier.predict(np.array(X_test)))]))
y_real  = encoder.inverse_transform([np.argmax(i, axis=None, out=None) for i in y_test])


# from sklearn.metrics import confusion_matrix
# from plot_confusion_matrix import plot_confusion_matrix

# cm = confusion_matrix(y_real, y_pred)

# plot_confusion_matrix(cm, target_names=['(0,0)','(0,1)','(0,2)','(1,0)','(1,1)','(1,2)','(2,0)','(2,1)','(2,2)']),
#                       title='Confusion matrix')

# =============================================================================
# Funcion de Prueba
# =============================================================================

history = model.fit(X_train, 
                    y_train,
                    epochs = best_parameters['epochs'],
                    batch_size = best_parameters['batch_size'],
                    verbose=0,
                    validation_data=(X_test,y_test))
plot_history(history)

full_multiclass_report(classifier,X_test,y_test,classes=['(0,0)','(0,1)','(0,2)','(1,0)','(1,1)','(1,2)','(2,0)','(2,1)','(2,2)'])

# =============================================================================
#     Move file to folder
# =============================================================================
mv = directory + '/1000a/Confusion_matrix.png'
os.system('mv Confusion_matrix.png '+ mv)

# =============================================================================
#< Plot loss and accuracy
# =============================================================================
from matplotlib import pyplot as plt

# ax1 = plt.subplot(121)
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('Model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='center right')
# plt.grid()
# #ax1.set_ylim([0.9, 1.02])

# plt.subplot(122)
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='center right')
# plt.grid()
# plt.subplots_adjust(wspace =0.4, hspace= 2.5)

history = classifier.fit(X_train, y_train, validation_split=0.33, epochs=150, batch_size=10, verbose=0)
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
        
plt.savefig( os.getcwd() + '/1000a/accuracy_over_epochs_train_1000.pdf')


# =============================================================================
#   Load Model
# =============================================================================
# load and evaluate a saved model
from numpy import loadtxt
from keras.models import load_model

# load model
classifier = load_model('model.h5')
# summarize model.
classifier.summary()
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
from fixrows  import fixrows
from merge_csv import fusionar_csv
from itertools import product
# =============================================================================
# Importing the dataset
# =============================================================================
a = list(product([100,250,350,500],['s','n']))

duration = 3 #segundos
f1 = 440    #término de procesos simples
f2 = 550    #Termino de grid search
f3 = 650    #Termino de cross_validation
f4 = 750    #Termino del codigo, paso al siguiente

# for i in range(len(a)):
  
#     j = a[i][0]
#     son = a[i][1].capitalize()
#     print(f"j = {a[i][0]}\nson = {a[i][1]}")

#     directory = os.getcwd()
#     print(f"directory = {directory+'/'}")

#     if os.path.isdir(os.getcwd() + '/{}_{}'.format(j,son)) == True:
#         pass
#     else:
#         os.mkdir(os.getcwd() + '/{}_{}'.format(j,son))
    
#     t = TicToc()
    
#     if son=='S':
#         df1 = fixrows('Potencia_r1').iloc[:j,:]
#         num_row = np.shape(df1)[0]
#         coords  = ['(1,0)' for j in range(num_row)]
#         coords = pd.DataFrame(coords,dtype=object, columns = ['X,Y'])
#         df1 = coords.join(df1, how='left')
#         df1.to_csv('Potencia_R1.csv')
         
#         df2 = fixrows('Potencia_r2').iloc[:j,:]
#         num_row = np.shape(df2)[0]
#         coords  = ['(0,0)' for j in range(num_row)]
#         coords = pd.DataFrame(coords,dtype=object, columns = ['X,Y'])
#         df2 = coords.join(df2, how='left')
#         df2.to_csv('Potencia_R2.csv')
         
#         df3 = fixrows('Potencia_r3').iloc[:j,:]
#         num_row = np.shape(df3)[0]
#         coords  = ['(0,1)' for j in range(num_row)]
#         coords = pd.DataFrame(coords,dtype=object, columns = ['X,Y'])
#         df3 = coords.join(df3, how='left')
#         df3.to_csv('Potencia_R3.csv')
        
#         df4 = fixrows('Potencia_r4').iloc[:j,:]
#         num_row = np.shape(df4)[0]
#         coords  = ['(1,1)' for j in range(num_row)]
#         coords = pd.DataFrame(coords,dtype=object, columns = ['X,Y'])
#         df4 = coords.join(df4, how='left')
#         df4.to_csv('Potencia_R4.csv')
          
#         # Fusionar archivos corregidos para obtener el archivo de potencias final
#         fusionar_csv('Potencia_R1','Potencia_R2','Potencia_R3','Potencia_R4')
         
#         df0 = fixrows('potencias_fusionado').iloc[:,1:]
#         os.system('rm Potencia_R*')
#         os.system('rm Potencia_r*_corregido.csv')
#         df0 = pd.read_csv('potencias_fusionado_corregido.csv').iloc[3:,1:]
#         os.system('play -nq -t alsa synth {} sine {}'.format(duration,f1))
#     else:
#         print("Leyendo archivo ya creado ")
#         df0 = pd.read_csv('potencias_fusionado_corregido.csv').iloc[3:,1:]
    
#     X = df0.iloc[:,1:].values #variables Dependientes (Potencias)
#     y = df0.iloc[:,0].values #values Independientes (Posición)
    
#     from keras.utils import np_utils
#     from sklearn.preprocessing import LabelEncoder
    
#     encoder = LabelEncoder()
#     encoder.fit(y)
#     y_encoded = encoder.transform(y)
#     y_encoded = np_utils.to_categorical(y_encoded)
    
#     # Splitting the dataset into the Training set and Test set
#     from sklearn.model_selection import train_test_split
#     X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size = 0.2, random_state = 0)
    
#     # =============================================================================
#     # Feature Scaling (Standarization)
#     # =============================================================================
#     if son == 'S':
#         print(son)
#         from sklearn.preprocessing import StandardScaler
#         sc = StandardScaler()
#         X_train = sc.fit_transform(X_train)
#         X_test = sc.transform(X_test)
#     else:
#         print(son)
#         # Feature Scaling (Normalization)
#         from sklearn import preprocessing
#         X_train = preprocessing.normalize(X_train)
#         X_test  = preprocessing.normalize(X_test)
#     # =============================================================================
#     # Buscar mejores parámetros
#     # =============================================================================
#     from keras.wrappers.scikit_learn import KerasClassifier
#     from sklearn.model_selection import GridSearchCV
#     from keras.models import Sequential
#     from keras.layers import Dropout
#     from keras.layers import Dense

#     print("Comenzando Grid_search\n")
#     t.tic()

#     def build_classifier(optimizer):
#         classifier = Sequential()
#         classifier.add(Dense(units = np.shape(X_test)[1]+1, kernel_initializer = 'uniform', activation = 'relu', input_dim = np.shape(X_test)[1]))
#         classifier.add(Dropout(rate = 0.2))
        
#         classifier.add(Dense(units = 24, kernel_initializer = 'uniform', activation = 'relu'))
#         classifier.add(Dropout(rate = 0.2))
        
#         classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu'))    
#         classifier.add(Dropout(rate = 0.2))
    
#         classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu'))
#         classifier.add(Dropout(rate = 0.2))
    
#         classifier.add(Dense(units = np.shape(y_test)[1], kernel_initializer = 'uniform', activation = 'softmax'))
#         classifier.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
#         return classifier
    
#     classifier = KerasClassifier(build_fn = build_classifier)
#     parameters = {'batch_size': [16,32,48],'epochs': [15, 25, 35],'optimizer': ['adam', 'adamax','rmsprop']}

#     grid = GridSearchCV(estimator = classifier,param_grid = parameters,
# #                           scoring = 'accuracy',
#                             cv = 3,n_jobs=-3)

#     grid_search_results = grid.fit(X_train, y_train)
    
#     best_parameters  = grid_search_results.best_params_
#     print(f"best_parameters = {grid_search_results.best_params_}")
#     print(f"best_accuracy =   {grid_search_results.best_score_}")
#     t.toc('Finalizado - Grid_search')
#     os.system('play -nq -t alsa synth {} sine {}'.format(duration,f2))
#     t1 = t.tocvalue()

#     means = grid_search_results.cv_results_['mean_test_score']
#     stds = grid_search_results.cv_results_['std_test_score']
#     params = grid_search_results.cv_results_['params']
    

# #     # =============================================================================
# #     # Cross Validation
# #     # =============================================================================
#     from keras.wrappers.scikit_learn import KerasClassifier
#     from sklearn.model_selection import cross_val_score
    
#     t.tic() 
#     print("\nEntrando en Cross Validation\n")
    
#     def build_classifier():
#         classifier = Sequential()
#         classifier.add(Dense(units = np.shape(X_test)[1]+1, kernel_initializer = 'uniform', activation = 'relu', input_dim = np.shape(X_test)[1]))
#         classifier.add(Dropout(rate = 0.2))
        
#         classifier.add(Dense(units = 24, kernel_initializer = 'uniform', activation = 'relu'))
#         classifier.add(Dropout(rate = 0.2))
        
#         classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu'))    
#         classifier.add(Dropout(rate = 0.2))
    
#         classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu'))
#         classifier.add(Dropout(rate = 0.2))
    
#         classifier.add(Dense(units = np.shape(y_test)[1], kernel_initializer = 'uniform', activation = 'softmax'))
#         classifier.compile(optimizer = best_parameters['optimizer'], loss = 'categorical_crossentropy', metrics = ['accuracy'])
#         return classifier
    
#     classifier = KerasClassifier(build_fn = build_classifier, batch_size = best_parameters['batch_size'], epochs = best_parameters['epochs'])
    
#     accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 5, n_jobs = -2)
#     ac = list(accuracies)
#     mean = accuracies.mean()
#     variance = accuracies.std()
#     t.toc('\nTiempo Cross-Validation: ')
#     os.system('play -nq -t alsa synth {} sine {}'.format(duration,f3))
#     time = t.tocvalue()
    
#     # =============================================================================
#     #     Distribución de probabilidad
#     # =============================================================================
#     import seaborn as sns
#     import matplotlib.pyplot as plt
        
#     print('>> Mean CV score is: ', round(np.mean(accuracies),3))
#     pltt = sns.distplot(pd.Series(accuracies,name='CV scores distribution'), color='r')
#     plt.savefig('CV_Accuracies_distribution.png')
#     plt.close()
    
#     # =============================================================================
#     #     Saving model
#     # =============================================================================
#     classifier = Sequential()
#     classifier.add(Dense(units = np.shape(X_test)[1]+1, kernel_initializer = 'uniform', activation = 'relu', input_dim = np.shape(X_test)[1]))
#     classifier.add(Dropout(rate = 0.2))
    
#     classifier.add(Dense(units = 24, kernel_initializer = 'uniform', activation = 'relu'))
#     classifier.add(Dropout(rate = 0.2))
    
#     classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu'))    
#     classifier.add(Dropout(rate = 0.2))
    
#     classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu'))    
#     classifier.add(Dropout(rate = 0.2))
    
#     classifier.add(Dense(units = np.shape(y_test)[1], kernel_initializer = 'uniform', activation = 'softmax'))
#     classifier.compile(optimizer = best_parameters['optimizer'], loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
#     # serialize model to JSON
#     model_json = classifier.to_json()
#     with open("model.json", "w") as json_file:
#         json_file.write(model_json)
#     # save model and architecture to single file
#     classifier.save(directory + "/{}_{}/model_{}.h5".format(j,son,j))
#     print("Saved model to disk\n")
    
#     # =============================================================================
#     #     Escritura de archivo
#     # =============================================================================
#     print(f"\nnp.shape(X_test)[0] = {np.shape(X_test)[0]}\n")
    
#     outFileName= directory + "/{}_{}/resultados_{}.txt".format(j,son,j)
#     f = open(outFileName,"w")
    
#     f.write("El número de elementos usados es: {}\n".format(j)+
#             "Los mejores parámetros son: {}".format(best_parameters) +
#             "\nTiempo de GridSearchCV  = {}".format(round(int(t1/60),2)) + 
#             "\nTiempo red neuronal  =  {}".format(round(int(time/60),2)) + 
#             "\nLa media obtenida es: {}".format(mean)+
#             "\nLa varianza obtenida es: {}".format(variance) + '\n'
#             )
#     for i in ac:
#         f.write("\tac: " + repr(round((i*100),2)) +"%" + '\n')

#     for i,j in enumerate(zip(params,means,stds)):
#         f.write("\nparams[{}] = {} --> means[{}] = {}\n".format(i,params[i],i,round(means[i],2)))
    
#     # =============================================================================
#     #                             Prediction
#     # =============================================================================
#     for i in range(1,15):
#         r = random.randint(0, np.shape(X_test)[0]-1)
#         y_pred = classifier.predict(np.array([X_test[r]]))
#         predictions = list(encoder.inverse_transform([np.argmax(y_pred, axis=None, out=None)]))
#         y_pred_prob = classifier.predict_proba(np.array([X_test[r]]))
#         f.write("\nFor the vector: ["+ repr(X_test[r])+ "]\t the predicted position is:" +  repr(predictions) +  "and its accuracy was:" + repr(round(np.amax(y_pred_prob),2)))
#     f.close()
    
#     print("Archivo escrito\n")
#     os.system('play -nq -t alsa synth {} sine {}'.format(duration,f1))
    
#     # =============================================================================
#     # Full multiclass report 
#     # =============================================================================
    
#     from plot_history import * 
#     from full_multiclass_report import * 
    
#     history = classifier.fit(X_train, 
#                         y_train,
#                         epochs = best_parameters['epochs'],
#                         batch_size = best_parameters['batch_size'],
#                         verbose=0,
#                         validation_data=(X_test,y_test))
#     plot_history(history)
    
#     full_multiclass_report(classifier,X_test,y_test,classes=['(0,0)','(0,1)','(1,0)','(1,1)'])
    
#     # =============================================================================
#     #     Move file to folder
#     # =============================================================================
#     print(f"directory = {directory}")
#     mv = directory + '/{}_{}/'.format(j,son)
#     os.system('mv Confusion_matrix.png '+ mv+ 'Confusion_matrix.png')
#     os.system('mv Loss.png '+ mv + 'Loss.png')
#     os.system('mv Accuracy.png '+ mv + 'Accuracy.png')
#     os.system('mv Classification_report.csv ' + mv + 'Classification_report.csv')
#     os.system('mv CV_Accuracies_distribution.png '+ mv+ 'CV_Accuracies_distribution.png')
#     os.system('mv model.json ' + mv + 'model_{}_{}.json'.format(j,son))
os.system('play -nq -t alsa synth {} sine {}'.format(duration,f4))
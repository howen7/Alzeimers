import matplotlib.pyplot as plt
import scipy
import numpy as np
import os 
import glob
import seaborn as sns

from PIL import Image
from scipy import ndimage


from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix

from keras import models
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.metrics import AUC
from sklearn.utils import compute_class_weight



def Vis_results(model,history, generator, samples, batch_size):
    Y_pred = model.predict_generator(generator, samples // batch_size +1) # so it lines up with the batches
    y_pred = np.argmax(Y_pred, axis=1)
    target_names = ['Mild','Moderate','None','Very Mild']
    print('-----------------------Val Classification Report-------------------')
    print(classification_report(generator.classes, y_pred, target_names=target_names))
    print('------------------------Val Confusion Matrix---------------------------')
    conf = confusion_matrix(generator.classes, y_pred, normalize='true')
    ax = sns.heatmap(conf, annot=True, xticklabels = target_names, yticklabels= target_names);
    ax.set_title('Test Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    
    
    fig, ax = plt.subplots(1, 3, figsize=(20, 3))
    ax = ax.ravel()

    for i, j in enumerate(['auc', 'loss', 'acc']):
        ax[i].plot(history.history[j])
        ax[i].plot(history.history['val_' + j])
        ax[i].set_title('Model {}'.format(j))
        ax[i].set_xlabel('epochs')
        ax[i].set_ylabel(j)
        ax[i].legend(['train', 'val'])
        
    return

def Vis_results_test(model,history, generator):
    Y_pred = model.predict_generator(generator) # so it lines up with the batches
    y_pred = np.argmax(Y_pred, axis=1)
    target_names = ['Mild','Moderate','None','Very Mild']
    print('-----------------------Test Classification Report-------------------')
    print(classification_report(generator.classes, y_pred, target_names=target_names))

    fontdict = {'weight' : 'bold',
            'size'   : 18}
    conf = confusion_matrix(generator.classes, y_pred, normalize='true')
    ax = sns.heatmap(conf, annot=True, xticklabels = target_names, yticklabels= target_names);
    ax.set_yticklabels(ax.get_yticklabels(), rotation = 0)
    ax.set_title('Test Confusion Matrix', fontdict = fontdict)
    ax.set_xlabel('Predicted', fontdict = fontdict)
    ax.set_ylabel('True', fontdict = fontdict)
    plt.savefig(f'../report/figures/Confusion_matrix_test', dpi = 300)
    
    return y_pred, Y_pred



def Vis_results2(model,history, generator, samples, batch_size):
    Y_pred = model.predict_generator(generator, samples // batch_size +1) # so it lines up with the batches
    y_pred = np.argmax(Y_pred, axis=1)
    target_names = ['Mild','Moderate','None','Very Mild']
    print('-----------------------Val Classification Report-------------------')
    print(classification_report(generator.classes, y_pred, target_names=target_names))
    fontdict = {'weight' : 'bold',
            'size'   : 18}
    conf = confusion_matrix(generator.classes, y_pred, normalize='true')
    ax = sns.heatmap(conf, annot=True, xticklabels = target_names, yticklabels= target_names);
    ax.set_yticklabels(ax.get_yticklabels(), rotation = 0)
    ax.set_title('Validation Confusion Matrix', fontdict = fontdict)
    ax.set_xlabel('Predicted', fontdict = fontdict)
    ax.set_ylabel('True', fontdict = fontdict)
    plt.savefig(f'../report/figures/Confusion_matrix_val', dpi = 300)
    
    fig, ax = plt.subplots(2, 1, figsize=(14, 10))
    ax = ax.ravel()

    for i, j in enumerate(['acc', 'loss']):
        ax[i].plot(history[j])
        ax[i].plot(history['val_' + j])
        ax[i].set_title(f'Model {j.title()}',  fontdict = fontdict)
        ax[i].set_xlabel('Epochs', fontdict = fontdict)
        ax[i].set_ylabel(j.title(), fontdict = fontdict)
        ax[i].legend(['train', 'val'])
        fig.tight_layout()
        plt.savefig(f'../report/figures/{j}', dpi = 300)
    return
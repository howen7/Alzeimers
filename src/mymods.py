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
    print('-----------------------Classification Report-------------------')
    print(classification_report(generator.classes, y_pred, target_names=target_names))
    print('------------------------Confusion Matrix---------------------------')
    conf = confusion_matrix(generator.classes, y_pred, normalize='true')
    ax = sns.heatmap(conf, annot=True, xticklabels = target_names, yticklabels= target_names);
    ax.set(xlabel='Predicted', ylabel='Actual')
    
    
    fig, ax = plt.subplots(1, 3, figsize=(20, 3))
    ax = ax.ravel()

    for i, j in enumerate(['auc', 'loss', 'acc']):
        ax[i].plot(history.history[j])
        ax[i].plot(history.history['val_' + j])
        ax[i].set_title('Model {}'.format(j))
        ax[i].set_xlabel('epochs')
        ax[i].set_ylabel(j)
        ax[i].legend(['train', 'val'])
        
    return y_pred

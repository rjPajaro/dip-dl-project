import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50, convnext, InceptionV3, EfficientNetB0
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report, roc_curve

def getResNet50(input_shape):
    model = ResNet50(include_top=True, weights=None, input_shape=input_shape, pooling = 'avg', classes = 1, classifier_activation="sigmoid")
    return model


def getConvNext(input_shape):
    model = convnext.ConvNeXtTiny(include_top=True, weights=None, input_shape=input_shape, pooling = 'avg', classes = 1, classifier_activation="sigmoid")
    return model

def getInceptionV3(input_shape):
    model = InceptionV3(include_top=True, weights=None, input_shape=input_shape, pooling = 'avg', classes = 1 , classifier_activation="sigmoid")
    return model

def getEfficientNetB0(input_shape):
    model = EfficientNetB0(include_top=True, weights=None, input_shape=input_shape, pooling = 'avg', classes = 1, classifier_activation="sigmoid")
    return model

def getConfusionMatrix(y_test, predictions):
    cm = confusion_matrix(y_test, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()
  

def getClassReport(y_test, predictions):
    print(classification_report(y_test, predictions))
    
def getLossVisual(history, epochs): #history => model.fit 
    loss_train = history.history['train_loss']
    loss_val = history.history['val_loss']
    plt.plot(epochs, loss_train, 'g', label='Training loss')
    plt.plot(epochs, loss_val, 'b', label='validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def getAccuracyVisual(history, epochs): #history => model.fit    
    loss_train = history.history['accuracy']
    loss_val = history.history['val_accuracy']
    plt.plot(epochs, loss_train, 'g', label='Training accuracy')
    plt.plot(epochs, loss_val, 'b', label='validation accuracy')
    plt.title('Training and Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    
def getROCScore(y_test, pred_model1, pred_model2, pred_model3):
    # roc curve for models
    fpr1, tpr1, thresh1 = roc_curve(y_test, pred_model1[:,1], pos_label=1)
    fpr2, tpr2, thresh2 = roc_curve(y_test, pred_model2[:,1], pos_label=1)
    fpr3, tpr3, thresh3 = roc_curve(y_test, pred_model3[:,1], pos_label=1)

    # roc curve for tpr = fpr 
    random_probs = [0 for i in range(len(y_test))]
    p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)
    
    plt.style.use('seaborn')

# plot roc curves
    plt.plot(fpr1, tpr1, linestyle='--',color='orange', label='Logistic Regression')
    plt.plot(fpr2, tpr2, linestyle='--',color='green', label='KNN')
    plt.plot(fpr3, tpr3, linestyle='--',color='red', label='Naives bayes')
    plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
    # title
    plt.title('ROC curve')
    # x label
    plt.xlabel('False Positive Rate')
    # y label
    plt.ylabel('True Positive rate')

    plt.legend(loc='best')
    plt.savefig('ROC',dpi=300)
    plt.show()
from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
from django.conf import settings
import os
import io
import base64
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import soundfile
import librosa
from sklearn.preprocessing import MinMaxScaler
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint 
import pickle
from keras.models import Sequential
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten, TimeDistributed, LSTM
from keras.layers import Conv2D
from keras.models import Sequential, load_model, Model
global username
global X_train, X_test, y_train, y_test, X, Y
accuracy = []
precision = []
recall = [] 
fscore = []
#function to calculate all metrics
def calculateMetrics(algorithm, y_test, predict):
    a = (accuracy_score(y_test,predict)*100)
    p = (precision_score(y_test, predict,average='macro') * 100)
    r = (recall_score(y_test, predict,average='macro') * 100)
    f = (f1_score(y_test, predict,average='macro') * 100)
    a = round(a, 3)
    p = round(p, 3)
    r = round(r, 3)
    f = round(f, 3)
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    return algorithm
def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
    sound_file.close()        
    return result
def values(filename, acc):
    f = open(filename, 'rb')
    train_values = pickle.load(f)
    f.close()
    accuracy_value = train_values[acc]
    return accuracy_value
X = np.load('model/X.npy')
Y = np.load('model/Y.npy')
X = np.reshape(X, (X.shape[0], (X.shape[1] * X.shape[2] * X.shape[3] * X.shape[4])))
scaler = MinMaxScaler((0,1))
X = scaler.fit_transform(X)
X = np.reshape(X, (X.shape[0], 1, 10, 6, 3))
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]
Y = to_categorical(Y)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and test
data = np.load("model/data.npy", allow_pickle=True)
X_train, X_test, y_train, y_test = data
cnn_lstm = Sequential()
#adding cnn layer with time distributed to extract spatial features
cnn_lstm.add(TimeDistributed(Conv2D(32, (3, 3), padding='same',activation = 'relu'), input_shape = (X.shape[1], X.shape[2], X.shape[3], X.shape[4])))
#max layer will collect relevant features from CNN
cnn_lstm.add(TimeDistributed(MaxPooling2D((4, 4))))
#dropout will remove ir-relevant features
cnn_lstm.add(Dropout(0.5))
#adding another layer to further optimize features
cnn_lstm.add(TimeDistributed(Conv2D(64, (3, 3), padding='same',activation = 'relu')))
cnn_lstm.add(TimeDistributed(MaxPooling2D((1, 1))))
cnn_lstm.add(Dropout(0.5))
cnn_lstm.add(TimeDistributed(Conv2D(128, (3, 3), padding='same',activation = 'relu')))
cnn_lstm.add(TimeDistributed(MaxPooling2D((1, 1))))
cnn_lstm.add(Dropout(0.5))
cnn_lstm.add(TimeDistributed(Conv2D(256, (2, 2), padding='same',activation = 'relu')))
cnn_lstm.add(TimeDistributed(MaxPooling2D((1, 1))))
cnn_lstm.add(Dropout(0.5))
cnn_lstm.add(TimeDistributed(Flatten()))
#adding LSTM layer to extract and optimzie temporal features  
cnn_lstm.add(LSTM(32))
cnn_lstm.add(Dense(units = y_train.shape[1], activation = 'softmax'))
#compiling, training and loading model
cnn_lstm.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
if os.path.exists("model/cnn_lstm.hdf5") == False:
    model_check_point = ModelCheckpoint(filepath='model/cnn_lstm.hdf5', verbose = 1, save_best_only = True)
    hist = cnn_lstm.fit(X_train, y_train, batch_size = 32, epochs = 50, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
    f = open('model/cnn_lstm_history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()    
else:
    cnn_lstm.load_weights("model/cnn_lstm.hdf5")
#performing prediction on test data    
predict = cnn_lstm.predict(X_test)
predict = np.argmax(predict, axis=1)
y_test1 = np.argmax(y_test, axis=1)
#call funcction to calculate accuracy and other metrics 
calculateMetrics("CNN + LSTM Algorithm", y_test1, predict)
conf_matrix = confusion_matrix(y_test1, predict)    
def Predict(request):
    if request.method == 'GET':
        return render(request, 'Predict.html', {})
def PredictAction(request):
    if request.method == 'POST':
        global scaler
        class_label = ['Fake', 'Original']
        myfile = request.FILES['t1'].read()
        name = request.FILES['t1'].name
        if os.path.exists('DeepFakeApp/static/'+name):
            os.remove('DeepFakeApp/static/'+name)
        with open('DeepFakeApp/static/'+name, "wb") as file:
            file.write(myfile)
        file.close()
        cnn_lstm = load_model("model/cnn_lstm.hdf5")
        mfcc = extract_feature('DeepFakeApp/static/'+name, mfcc=True, chroma=True, mel=True)
        mfcc = np.reshape(mfcc, (10, 6, 3))
        test = []
        test.append([mfcc])
        test = np.asarray(test)
        test = np.reshape(test, (test.shape[0], (test.shape[1] * test.shape[2] * test.shape[3] * test.shape[4])))
        test = scaler.transform(test)
        test = np.reshape(test, (test.shape[0], 1, 10, 6, 3))
        predict = cnn_lstm.predict(test)
        predict = np.argmax(predict)
        predict = class_label[predict]
        context= {'data':'<font size="3" color="blue">Uploaded Audio Predicted As : '+predict+'</font>'}
        return render(request, 'UserScreen.html', context)
def TrainModel(request):
    if request.method == 'GET':
        global X_train, X_test, y_train, y_test
        global accuracy, precision, recall, fscore, conf_matrix
        class_label = ['Fake', 'Original']
        output='<table border=1 align=center width=100%><tr><th><font size="3" color="black">Algorithm Name</th><th><font size="3" color="black">Accuracy</th>'
        output += '<th><font size="3" color="black">Precision</th><th><font size="3" color="black">Recall</th><th><font size="3" color="black">FSCORE</th></tr>'
        algorithms = ['CNN + LSTM Algorithm']
        for i in range(len(algorithms)):
            output += '<td><font size="3" color="black">'+algorithms[i]+'</td><td><font size="3" color="black">'+str(accuracy[i])+'</td><td><font size="3" color="black">'+str(precision[i])+'</td>'
            output += '<td><font size="3" color="black">'+str(recall[i])+'</td><td><font size="3" color="black">'+str(fscore[i])+'</td></tr>'
        output+= "</table></br>"
        arch1_acc = values("model/cnn_lstm_history.pckl", "accuracy")
        index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
                 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
        figure, axis = plt.subplots(nrows=1, ncols=2,figsize=(10, 3))#display original and predicted segmented image
        axis[0].set_title("Confusion Matrix Prediction Graph")
        axis[1].set_title("CNN Training Accuracy Graph")
        ax = sns.heatmap(conf_matrix, xticklabels = class_label, yticklabels = class_label, annot = True, cmap="viridis" ,fmt ="g", ax=axis[0]);
        ax.set_ylim([0,len(class_label)])    
        axis[1].plot(index, arch1_acc, color="green")
        axis[1].legend(['Training Accuracy'], loc='lower right')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        #plt.close()
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        plt.clf()
        plt.cla()
        context= {'data':output, 'img': img_b64}
        return render(request, 'UserScreen.html', context)
def LoadDataset(request):    
    if request.method == 'GET':
        global X_train, X_test, y_train, y_test, X, Y
        class_label = ['Fake', 'Original']
        output = '<font size="3" color="black">Deep Fake Audio Dataset Loaded</font><br/>'
        output += '<font size="3" color="blue">Total Audio MFCC Features Extracted from each Audio = 180</font><br/>'
        output += '<font size="3" color="blue">Different Class Labels found in Dataset = '+str(class_label)+'</font><br/><br/>'
        output += '<font size="3" color="black">Dataset Train & Test Split details</font><br/>'
        output += '<font size="3" color="blue">80% dataset MFCC features used to train Algorithm = '+str(X_train.shape[0] * 180)+'</font><br/>'
        output += '<font size="3" color="blue">20% dataset MFCC features used to test Algorithm = '+str(X_test.shape[0] * 180)+'</font><br/>'
        context= {'data':output}
        return render(request, 'UserScreen.html', context)
def UserLoginAction(request):
    global username
    if request.method == 'POST':
        global username
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        if username == 'admin' and password == 'admin':
            context= {'data':'Welcome '+username}
            return render(request, "UserScreen.html", context)
        else:
            context= {'data':'Invalid username'}
            return render(request, 'UserLogin.html', context)
def UserLogin(request):
    if request.method == 'GET':
       return render(request, 'UserLogin.html', {})
def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})

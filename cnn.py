# module imports
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import random

# model imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# processing imports
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


import tensorflow as tf
from tensorflow.keras.utils import to_categorical

from keras.layers import Dense, LSTM, MaxPool1D, Flatten, Dropout # importing dense layer
from keras.models import Sequential #importing Sequential layer
from keras.layers import Input
from keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D







# fetch the training file
file_path_20_percent = './NSL-KDD/KDDTrain+_20Percent.txt'
file_path_full_training_set = './NSL-KDD/KDDTrain+.txt'
file_path_test = './NSL-KDD/KDDTest+.txt' 

#df = pd.read_csv(file_path_20_percent)
df = pd.read_csv(file_path_full_training_set)
test_df = pd.read_csv(file_path_test)

# add the column labels
columns = (['duration'
,'protocol_type'
,'service'
,'flag'
,'src_bytes'
,'dst_bytes'
,'land'
,'wrong_fragment'
,'urgent'
,'hot'
,'num_failed_logins'
,'logged_in'
,'num_compromised'
,'root_shell'
,'su_attempted'
,'num_root'
,'num_file_creations'
,'num_shells'
,'num_access_files'
,'num_outbound_cmds'
,'is_host_login'
,'is_guest_login'
,'count'
,'srv_count'
,'serror_rate'
,'srv_serror_rate'
,'rerror_rate'
,'srv_rerror_rate'
,'same_srv_rate'
,'diff_srv_rate'
,'srv_diff_host_rate'
,'dst_host_count'
,'dst_host_srv_count'
,'dst_host_same_srv_rate'
,'dst_host_diff_srv_rate'
,'dst_host_same_src_port_rate'
,'dst_host_srv_diff_host_rate'
,'dst_host_serror_rate'
,'dst_host_srv_serror_rate'
,'dst_host_rerror_rate'
,'dst_host_srv_rerror_rate'
,'attack'
,'level'])

df.columns = columns
test_df.columns = columns

# map normal to 0, all attacks to 1
is_attack = df.attack.map(lambda a: 0 if a == 'normal' else 1)
test_attack = test_df.attack.map(lambda a: 0 if a == 'normal' else 1)

#data_with_attack = df.join(is_attack, rsuffix='_flag')
df['attack_flag'] = is_attack
test_df['attack_flag'] = test_attack

# lists to hold our attack classifications
dos_attacks = ['apache2','back','land','neptune','mailbomb','pod','processtable','smurf','teardrop','udpstorm','worm']
probe_attacks = ['ipsweep','mscan','nmap','portsweep','saint','satan']
privilege_attacks = ['buffer_overflow','loadmdoule','perl','ps','rootkit','sqlattack','xterm']
access_attacks = ['ftp_write','guess_passwd','http_tunnel','imap','multihop','named','phf','sendmail','snmpgetattack','snmpguess','spy','warezclient','warezmaster','xclock','xsnoop']

# we will use these for plotting below
attack_labels = ['Normal','DoS','Probe','Privilege','Access']

# helper function to pass to data frame mapping
def map_attack(attack):
    if attack in dos_attacks:
        # dos_attacks map to 1
        attack_type = 1
    elif attack in probe_attacks:
        # probe_attacks mapt to 2
        attack_type = 2
    elif attack in privilege_attacks:
        # privilege escalation attacks map to 3
        attack_type = 3
    elif attack in access_attacks:
        # remote access attacks map to 4
        attack_type = 4
    else:
        # normal maps to 0
        attack_type = 0
        
    return attack_type

# map the data and join to the data set
attack_map = df.attack.apply(map_attack)
df['attack_map'] = attack_map

test_attack_map = test_df.attack.apply(map_attack)
test_df['attack_map'] = test_attack_map

le=LabelEncoder()
clm=['protocol_type', 'service', 'flag', 'attack']
for x in clm:
    df[x]=le.fit_transform(df[x])
    test_df[x]=le.fit_transform(test_df[x])

print(df)

x_train=df.drop('attack_flag',axis=1)
x_train=x_train.drop('attack',axis=1)
x_train=x_train.drop('attack_map',axis=1)
x_train=x_train.drop('level',axis=1)
y_train=df[['attack_flag']]



X_train, X_test, Y_train, Y_test = train_test_split(x_train,y_train, test_size=0.20, random_state=42)

# max_features = 43
# embedding_size = 20

# Convolution
kernel_size = 5
filters = 64
pool_size = 4

# LSTM
lstm_output_size = 70

# Training
batch_size = 30
epochs = 2

X_train = np.array(X_train)
X_test = np.array(X_test)
#cnn-input
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#lstm-input
# X_train = np.reshape(X_train, ( X_train.shape[0], 1 , X_train.shape[1] ))
Y_train = np.array(Y_train)

# X_test = np.array(x_test)
# X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
# Y_test = np.array(y_test)

model = Sequential() # initializing model
#lstm
# model.add(LSTM(64,return_sequences=True,input_shape = (1, X_train.shape[2])))
# model.add(Dropout(0.2))
# model.add(LSTM(64,return_sequences=True))
# model.add(Dropout(0.2))
# model.add(LSTM(64,return_sequences=True))
# model.add(Flatten())
# model.add(Dense(units=1))
#cnn
# model.add(Conv1D(32, 9, padding="same",input_shape = (X_train.shape[1], 1), activation='relu'))
# model.add(MaxPool1D(pool_size=(2)))
# model.add(Dropout(0.2))
# model.add(Conv1D(32, 3, padding="same", activation='relu'))
# model.add(MaxPool1D(pool_size=(4)))  
# model.add(Dropout(0.2))
# model.add(Dense(units=1))

# cnn-lstm
model.add(Conv1D(32, 9, padding="same",input_shape = (X_train.shape[1], 1), activation='relu'))
model.add(MaxPool1D(pool_size=(2)))
# model.add(LSTM(units=16,return_sequences=False,dropout=0.2))
model.add(Dense(units=1))
# end
# model.add(LSTM(64,return_sequences=True))
# model.add(Dropout(0.2))
# model.add(LSTM(64,return_sequences=True))
# model.add(Flatten())
# model.add(LSTM(units=512,return_sequences=False,dropout=0.2))
# model.add(Activation('sigmoid'))
# output layer with softmax activation
# model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=50, batch_size=250)
model.save('Chlee')

# 20% train data test
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
Y_test = np.array(Y_test)

test_results = model.evaluate(X_test, Y_test, verbose=1)

# no trained data test
# x_test=test_df.drop('attack_flag',axis=1)
# x_test=test_df.drop('attack',axis=1)
# x_test=test_df.drop('attack_map',axis=1)
# x_test=test_df.drop('level',axis=1)
# X_test2 = np.array(x_test)
# X_test2 = np.reshape(X_test2, (X_test2.shape[0], X_test2.shape[1], 1))
# y_test=test_df[['attack_flag']]
# Y_test2 = np.array(y_test)

# test_results = model.evaluate(X_test2, Y_test2, verbose=1)
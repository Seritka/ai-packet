# module imports
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, accuracy_score, precision_recall_curve, roc_curve
import joblib
import tensorflow as tf

np.random.seed(2)



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

x_train=df.drop('attack_flag',axis=1)
x_train=x_train.drop('attack',axis=1)
x_train=x_train.drop('attack_map',axis=1)
x_train=x_train.drop('level',axis=1)
y_train=df[['attack_flag']]

X_train, X_test, Y_train, Y_test = train_test_split(x_train,y_train, test_size=0.20, random_state=42)


std = StandardScaler()

x_train_scaled = std.fit_transform(X_train)
x_test_scaled = std.transform(X_test)

svm = SVC(kernel='poly', gamma=0.5, C=5.0, verbose=True)

svm.fit(x_train_scaled, Y_train.values.ravel())
joblib.dump(svm,'svm.pkl')
model = joblib.load('svm.pkl')

print("크기:", x_train_scaled.shape, x_test_scaled.shape)
#prediction = model.predict(x_test_scaled)

print("훈련 정확도:", round(model.score(x_train_scaled, Y_train), 4))
print("테스트 정확도:", round(model.score(x_test_scaled, Y_test), 4))
print("성능 평가 지표:", accuracy_score(Y_test, model.predict(x_test_scaled)) * 100)

"""
feature들의 상관 관계 분석
plt.figure(figsize=(10,8))
import seaborn
seaborn.heatmap(df.corr(method='spearman'), annot=True)
plt.show()
"""
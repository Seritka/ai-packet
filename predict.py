import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

model = joblib.load('svm.pkl')

texts = ['0,udp,private,SF,44,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,0.00,0.00,0.00,0.00,1.00,0.00,0.00,255,254,1.00,0.01,0.00,0.00,0.00,0.00,0.00,0.00,snmpguess,16'.split(',')
, '13,tcp,telnet,SF,118,2425,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,0.00,0.00,0.00,0.00,1.00,0.00,0.00,26,10,0.38,0.12,0.04,0.00,0.00,0.00,0.12,0.30,guess_passwd,2'.split(',')]
for text in texts:
    df = pd.DataFrame([text])
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

    le=LabelEncoder()
    clm=['protocol_type', 'service', 'flag', 'attack']
    for x in clm:
        df[x]=le.fit_transform(df[x])

    x_train=df.drop('attack',axis=1)
    x_train=x_train.drop('attack_map',axis=1)
    x_train=x_train.drop('level',axis=1)

    print("악성 패킷:", False if model.predict(x_train.values) == 0 else True)
import soundfile as sf
import io
from six.moves.urllib.request import urlopen
import librosa
import pandas as pd
import tqdm
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten
from tensorflow.keras.optimizers import Adam
from sklearn import metrics
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime 
import math
import pyloudnorm as pyln
import csv
from sklearn.metrics import accuracy_score
import cleanlab
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from cleanlab.classification import CleanLearning
from cleanlab.filter import find_label_issues
from pyvad import vad, trim, split
import librosa
import matplotlib.pyplot as plt
import IPython.display
import requests
import ffmpeg
import csv
import os


def signaltonoise(Arr, axis=0, ddof=0):
    Arr = np.asanyarray(Arr)
    me = Arr.mean(axis)
    sd = Arr.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, me/sd)
array2 = [50, 12, 15, 34, 5] 
print(signaltonoise(array2,axis=0,ddof=0))

def after_vad(data,fs):
    data = np.hstack((data, -data))
    data *=0.95 / np.abs(data).max()
    edges = split(data, fs, fs_vad = 8000, hop_length = 10, vad_mode=3)
    new = []

    for i, edge in enumerate(edges):
        seg = data[edge[0]:edge[1]]
        new = np.concatenate((new,seg))

    return new,fs
    
hop_length = 256
frame_length = 512
sr = 8000
meter = pyln.Meter(sr) # create BS.1770 meter

import numpy as np

def features_extractor(audio, sample_rate):
    # print(type(audio))
    audio = np.array(audio)
    rolloff_min = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sample_rate, roll_percent=0.01))
    flatness = np.mean(librosa.feature.spectral_flatness(y=audio))
    spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate))
    cent = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sample_rate))
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
    rmse = np.mean(librosa.feature.rms(audio, frame_length=frame_length, hop_length=hop_length, center=True))
    loudness = meter.integrated_loudness(audio) # measure loudness
    snr = signaltonoise(audio,axis=0,ddof=0)
    
    return float(rolloff_min),float(flatness),float(spec_bw),float(cent),float(zcr),float(rmse),float(loudness),float(snr)

def get_features(save_file, *, list_file):
    header = ['audio', 'rolloff_min','flatness','spec_bw','cent','zcr','rmse','loudness','snr']
    if not os.path.exists(save_file):
        with open(save_file, 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            files_ok = []
    else:
        files_ok = list(pd.read_csv(save_file)['audio'])

    print('start get file need to get features')

    print(len(list_file))
    dict_order = {list_file[i] : i for i in tqdm.tqdm(range(len(list_file)))}
    dict_name = {get_name(_) : _ for _ in tqdm.tqdm(list_file)}
    print(len(dict_order), len(dict_name))
    list_name_no_dup = list(set(dict_name.keys()) - set(files_ok))
    list_file = sorted([dict_name[name] for name in list_name_no_dup], key = lambda x : dict_order[x])
    
    del dict_order
    del dict_name
    del list_name_no_dup
    print(list_file[:2])
    del files_ok
    gc.collect()

    print('start get features')

    with open(save_file, 'a', encoding='UTF8') as f1:
        writer1 = csv.writer(f1)

        for record_audio in tqdm.tqdm(list_file):
            filename = get_name(record_audio)
            
            try:
                audio, fs = librosa.load(record_audio,sr=8000)
            except:
                print('LOAD ERROR')
                writer1.writerow([filename] + ['LOAD ERROR'] * (len(header) - 1))
                continue

            audio, sample_rate = after_vad(audio,fs)
            # print('DONE')
            if len(audio) == 0:
                print(len(audio))
                writer1.writerow([filename] + ['SILENCE'] * (len(header) - 1))
                continue

            rolloff_min,flatness,spec_bw,cent,zcr,rmse,loudness,snr = features_extractor(audio, sample_rate)
            writer1.writerow([filename,rolloff_min,flatness,spec_bw,cent,zcr,rmse,loudness,snr])

import gc

get_name = lambda link : link.split('/')[-1]

save_file = '/home2/ivr-mappingcode/data/audio_features_check_thang_4.csv'

with open('/home2/ivr-mappingcode/huynv/save/found_thang_4.txt', 'r') as f:
    lines = f.readlines()

list_file = [line.strip() for line in lines]

del lines
gc.collect()

get_features(save_file, list_file = list_file)
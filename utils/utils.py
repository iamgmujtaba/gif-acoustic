from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from urllib.request import Request, urlopen
import numpy as np
from PIL import Image
import json as simplejson

from scipy.io import wavfile
import pandas  as pd
import crepe
import itertools
import subprocess
from os import system
import soundfile as sf
from pydub import AudioSegment

import csv
import re
import operator 
import time
import os
import glob
import shutil
import requests
import librosa


####################################################################
####################################################################

def print_config(config):
    print('#'*60)
    print('Training configuration:')
    for k,v  in vars(config).items():
        print('  {:>20} {}'.format(k, v))
    print('#'*60)

def write_config(config, json_path):
    with open(json_path, 'w') as f:
        f.write(simplejson.dumps(vars(config), indent=4, sort_keys=True))

def output_subdir(config):
    prefix = time.strftime("%Y_%m_%d_%H%M")
    subdir = "{}_{}_{}".format(prefix, config.dataset, config.model)
    return os.path.join(config.save_dir, subdir)

def prepare_output_dirs(config):
    # Set output directories
    config.save_dir = output_subdir(config)
    config.checkpoint_dir = os.path.join(config.save_dir, 'checkpoints')
    config.log_dir = os.path.join(config.save_dir, 'logs')

    # And create them
    if os.path.exists(config.save_dir):
        # Only occurs when experiment started the same minute
        shutil.rmtree(config.save_dir)

    os.mkdir(config.save_dir)
    os.mkdir(config.checkpoint_dir)
    os.mkdir(config.log_dir)
    return config

####################################################################
####################################################################
#create directory for hls server files to download
def output_walter_subdir(config, movie_id):
    prefix = time.strftime("%Y_%m_%d") 
    subdir = "{}_{}".format(prefix, movie_id)
    return os.path.join(config.video_path, subdir)

def prepare_walter_dirs(config, movie_id):
    # Set output directories
    config.video_path = output_walter_subdir(config, movie_id)
    config.segments_dir = os.path.join(config.video_path, 'segments')
    config.audio_dir = os.path.join(config.video_path, 'audio')
    config.climax_dir = os.path.join(config.video_path, 'climax')
    config.gif_dir = os.path.join(config.video_path, 'gif')

    # And create them
    if os.path.exists(config.video_path):
        # Only occurs when experiment started the same minute
        shutil.rmtree(config.video_path)

    os.mkdir(config.video_path)
    os.mkdir(config.segments_dir)
    os.mkdir(config.audio_dir)
    os.mkdir(config.climax_dir)
    os.mkdir(config.gif_dir)
    
    return config

####################################################################
####################################################################
def load_audio(filepath, song_samples = 660000):
    y, sr = librosa.load(filepath)
    y = y[:song_samples]
    return y, sr

####################################################################
####################################################################
def split_audio(X, window = 0.1, overlap = 0.5):
    # Empty lists to hold our results
    temp_X = []
    # Get the input song array size
    xshape = X.shape[0]
    chunk = int(xshape*window)
    offset = int(chunk*(1.-overlap))
    
    # Split the song and create new ones on windows
    spsong = [X[i:i+chunk] for i in range(0, xshape - chunk + offset, offset)]
    for s in spsong:
        if s.shape[0] != chunk:
            continue

        temp_X.append(s)
    return np.array(temp_X)

####################################################################
####################################################################
def to_melspec(signals):
    # mel_spec = [librosa.feature.melspectrogram(i) for i in signals]
    melspec = lambda x: librosa.feature.melspectrogram(x, sr=22050, n_fft=1024, hop_length=512)[:, :, np.newaxis]
    spec_array = map(melspec, signals)
    return np.array(list(spec_array))
    
####################################################################
####################################################################
#download audio file
def get_file_walter(movie_url, dest_path):
    file_name = movie_url.split('/')[-1]

    u = urlopen(movie_url)
    f = open(os.path.join(dest_path,file_name), 'wb')
    resp = requests.get(movie_url)
    
    file_size = int(resp.headers['content-length'])

    file_size_dl = 0
    block_sz = 8192
    while True:
        buffer = u.read(block_sz)
        if not buffer:
            break

        file_size_dl += len(buffer)
        f.write(buffer)
        status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)

####################################################################
####################################################################

def get_entire_audio_frequency(input_path):
    audio_file = os.path.join(input_path, 'audio', 'input_audio.wav')
    sr, audio = wavfile.read(audio_file)
    time, frequency, confidence, activation = crepe.predict(audio, sr, viterbi=True)

    csv_file = os.path.join(input_path,'max_freq.csv')
    f_data = np.vstack([time, frequency]).transpose()
    np.savetxt(csv_file,f_data, fmt= ['%.20f','%.20f'], delimiter= ',',header= 'time,frequency', comments='')

    df = pd.read_csv(csv_file)
    gh = df.loc[df['frequency'].idxmax()]
    gh.to_csv(csv_file)

####################################################################
####################################################################
def get_climax_audio_frequency(input_path):
    #extract climax from audio file
    audio_file = os.path.join(input_path, 'audio', 'input_audio.wav')
    extract_audio_climax(audio_file, os.path.join(input_path, 'climax'))
    
    climax_audio_file = os.path.join(input_path, 'climax', 'climax_part.wav')
    
    sr_2, audio_2 = wavfile.read(climax_audio_file)
    time_2, frequency_2, confidence_2, activation_2 = crepe.predict(audio_2, sr_2, viterbi=True)

    csv_climax_file = os.path.join(input_path,'max_climax_freq.csv')
    f_data_2 = np.vstack([time_2, frequency_2]).transpose()
    np.savetxt(csv_climax_file,f_data_2, fmt= ['%.20f','%.20f'], delimiter= ',',header= 'time,frequency', comments='')

    df_2 = pd.read_csv(csv_climax_file)
    gh_2 = df_2.loc[df_2['frequency'].idxmax()]
    gh_2.to_csv(csv_climax_file)


####################################################################
####################################################################
def get_freq_timestamp(input_file):
    row_number = 1
    column_number = 1
    
    with open(input_file, 'r') as f:
        row = next(itertools.islice(csv.reader(f), row_number, row_number+1))
        cell_value = row[column_number]
        timestamp = round(float(cell_value))
        return timestamp

####################################################################
####################################################################
def convert_ts_mp4(input_path, segment_name):
    file_name, file_ext = os.path.splitext(os.path.join(input_path,segment_name))
    command2 = 'ffmpeg -i '+ file_name+file_ext +' -acodec copy -vcodec copy '+ file_name +'.mp4'
    print(command2)    
    system(command2)

####################################################################
####################################################################
def generate_GIF(video_path, segment_name, gif_name):
    input_path = os.path.join(video_path, 'segments')
    file_name, file_extension = os.path.splitext(segment_name)
    input_mp4 = os.path.join(input_path, file_name+'.ts')
    output_dir = os.path.join(video_path, 'gif')

    pal_cmd = ("ffmpeg -y -t 4 -i " + input_mp4 + " -vf fps=10,scale=320:-1:flags=lanczos,palettegen " + os.path.join(output_dir, gif_name+file_name)+"_palette.png")
    gif_cmd = ("ffmpeg -y -t 4 -i " + input_mp4 + " -i "+ os.path.join(output_dir, gif_name+file_name) +"_palette.png " +" -lavfi \"fps=15,scale=320:-1:flags=lanczos[x];[x][1:v]paletteuse\" " + os.path.join(output_dir, gif_name+file_name) +"_max_freq.gif")

    print(pal_cmd)
    print(gif_cmd)

    subprocess.call(pal_cmd, shell=True)
    subprocess.call(gif_cmd, shell=True)

####################################################################
####################################################################
def generate_webp(input_path, segment_name, gif_name):
    file_name, file_extension = os.path.splitext(segment_name)
    input_mp4 = os.path.join(input_path, 'segments', segment_name)
    output_dir = os.path.join(input_path, 'gif')

    webp_cmd = ("ffmpeg -y -ss 1.0 -t 3 -i " + input_mp4 + " -vcodec libwebp -lossless 1 -preset default -loop 0 -an -vsync 0 " + os.path.join(output_dir, gif_name+file_name)+"_max_freq.webp")
    subprocess.call(webp_cmd, shell=True)

####################################################################
####################################################################
def get_climax_duration(input_path):
    wav_file = os.path.join(input_path, 'audio', 'input_audio.wav')
    audio_file = sf.SoundFile(wav_file)
    file_length = (len(audio_file) /audio_file.samplerate)
    #Get 25% length for climax part 
    total_climax_length = ((len(audio_file) /audio_file.samplerate) * 0.75)

    return total_climax_length
    
####################################################################
####################################################################
def extract_audio_climax(input_path, output_dir):
    # wav_file = os.path.join(input_path, 'input_audio.wav')
    wav_file = input_path

    audio_file = sf.SoundFile(wav_file)
    file_length = (len(audio_file) /audio_file.samplerate)
    #print("Time sec: ",file_length)    

    #Get 25% length for climax part 
    total_climax_length = ((len(audio_file) /audio_file.samplerate) * 0.75)
    # print("Starting Point",total_climax_length)

    total_climax_audio = AudioSegment.from_wav(wav_file)
    climax_part = file_length - total_climax_length

    twenty_five_climax = total_climax_audio[-climax_part *1000:]
    
    #more 5% remove from climax part
    acc_climax_part = (climax_part * 0.45)
    acc = climax_part - acc_climax_part

    acc_climax_audio = twenty_five_climax[:acc *1000]
    acc_climax_audio.export(os.path.join(output_dir, 'climax_part.wav'),format = "wav")

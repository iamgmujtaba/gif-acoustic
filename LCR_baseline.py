from keras.models import load_model
import os
import librosa
import numpy as np
import time
from subprocess import call

from utils.utils import load_audio, split_audio, to_melspec
from utils.utils import get_file_walter, get_freq_timestamp, get_climax_duration
from utils.utils import get_entire_audio_frequency

from utils.SGDW import SGDW
from utils.utils import prepare_walter_dirs

from config import parse_opts

genres = {0: 'metal', 1 : 'disco', 2: 'classical', 3 :'hiphop', 4:'jazz', 
          5 :'country', 6:'pop', 7:'blues', 8:'reggae', 9:'rock'}

vid1 = 'David.Guetta.Titanium/'
vid2 = 'Ed.Sheeran.Perfect/'
vid3 = 'Shape.of.You/'
vid4 = 'Enrique.Iglesias.SUBEME/'
vid5 = 'Gorillaz.Clint.Eastwood/'
vid6 = 'Imagine.Dragons.Thunder/'
vid7 = 'Luis.Fonsi.Despacito/'
vid8 = 'Maroon.5.Sugar/'
vid9 = 'Marshmello.Bastille.Happier/'
vid10 = 'Michael.Jackson.They.Don/'
vid11 = 'MICHAEL.JACKSON.Thriller/'
vid12 = 'Pharrell.Williams.Happy/'
vid13 = 'PSY.DADDY/'
vid14 = 'PSY.GANGNAM.STYLE/'
vid15 = 'Shakira.Waka.Waka/'
vid16 = 'TONES.AND.I.DANCE/'


song_list = [vid1, vid2, vid3, vid4, vid5, vid6, vid7, vid8, vid9,
            vid10, vid11, vid12, vid13, vid14, vid15, vid16]

model_path = './output/2020_06_02_1528_gtzan_cnn/checkpoints/042-0.25.hdf5'
model = load_model(model_path, custom_objects={'SGDW': SGDW})

music_genre = ''
freq_timestamp = ''
####################################################################
####################################################################

def download_video():
    get_file_walter(video_url, config.video_path)

def extract_audio():
    cmd = 'ffmpeg -i ' + input_video + ' -vn -acodec pcm_s16le -ar 44100 -ac 2 ' + out_audio_dir + '/input_audio.wav'
    print(cmd)
    call(cmd,shell=True)

def identify_genre():
    global music_genre 
    spectro = []

    y = load_audio(audio_file)[0]

    signals = split_audio(y)
    spec_array = to_melspec(signals)

    spectro.extend(spec_array)
    spectro = np.array(spectro)
    spectro = np.squeeze(np.stack((spectro,)*3,-1))

    predictions = np.array(model.predict(spectro))
    preds = np.argmax(predictions, axis=1)

    print('-'*40)
    music_genre = genres[np.bincount(preds).argmax()]
    print(music_genre)


def estimate_pitch():    
    global freq_timestamp
    #cmd = 'ffmpeg -i '+ audio_file +' -acodec pcm_s16le -ac 1 -ar 16000 '+ out_audio_dir +'/input_audio.wav'
    #print(cmd)
    #call(cmd,shell=True)
    
    get_entire_audio_frequency(config.video_path)
    #Identify the timestamp for the max frequency
    freq_timestamp = get_freq_timestamp(max_freq_csv)


def create_GIF(video_path, output_dir,  gif_name, start_time):
    name =  gif_name + 'video'
    file_name, file_extension = os.path.splitext(name)
    print('start_time',start_time)

    pal_cmd = ("ffmpeg -y -ss " + str(start_time) + " -t 3 -i " + video_path + " -vf fps=10,scale=320:-1:flags=lanczos,palettegen " + os.path.join(output_dir, file_name)+"_palette.png")
    gif_cmd = ("ffmpeg -y -ss " + str(start_time) + " -t 3 -i " + video_path + " -i "+ os.path.join(output_dir, file_name) +"_palette.png " +" -lavfi \"fps=15,scale=320:-1:flags=lanczos[x];[x][1:v]paletteuse\" " + os.path.join(output_dir, file_name) +"_max_freq.gif")

    call(pal_cmd, shell=True)
    call(gif_cmd, shell=True)

def main(video_dir):
    text_file = open(os.path.join(video_dir,'process.txt'), "w")
    text_file.write(video_dir)
    text_file.write('\n')
    
    start1 = time.time()
    download_video()
    end1 = time.time()
    text_file.write('Download Video sec: ' + str(round(end1 - start1, 2)) + ' mint: ' + str(round (end1 - start1, 2)/60) )
    text_file.write('\n')

    start2 = time.time()
    extract_audio()
    end2 = time.time()
    text_file.write('Extract Audio sec: ' + str(round(end2 - start2, 2)) + ' mint: ' + str(round (end2 - start2, 2)/60) )
    text_file.write('\n')
    
    start3 = time.time()
    identify_genre()
    end3 = time.time()
    text_file.write('Idenfity Genre sec: ' + str(round(end3 - start3, 2)) + ' mint: ' + str(round (end3 - start3, 2)/60) )
    text_file.write('\n')

    start4 = time.time()
    estimate_pitch()
    end4 = time.time()
    text_file.write('Estimate Pitch sec: ' + str(round(end4 - start4, 2)) + ' mint: ' + str(round (end4 - start4, 2)/60) )
    text_file.write('\n')

    start5 = time.time()
    create_GIF(input_video, out_gif_dir, 'full_', freq_timestamp)
    end5 = time.time()
    text_file.write('Generate GIF sec: ' + str(round(end5 - start5, 2)) + ' mint: ' + str(round (end5 - start5, 2)/60) )
    text_file.write('\n')
    text_file.write('\n')
    text_file.write('Music Genre :' + music_genre)
    text_file.close()

####################################################################
####################################################################


if __name__ == "__main__":
    
    count = 0
    while count < len(song_list):
        print('='*60)
        print(song_list[count])
        selected_music = song_list[count]
        count += 1

        config = parse_opts()
        config = prepare_walter_dirs(config, selected_music)
        
        audio_url = os.path.join(config.walter_ip, selected_music, 'audio', 'input_audio.wav')
        video_url = os.path.join(config.walter_ip, selected_music, 'input_resize_480_360.mp4')

        audio_file = os.path.join(config.video_path, 'audio', 'input_audio.wav')
        input_video = os.path.join(config.video_path, 'input_resize_480_360.mp4')
        out_audio_dir = os.path.join(config.video_path, 'audio')
        out_gif_dir = os.path.join(config.video_path, 'gif')

        max_freq_csv = os.path.join(config.video_path,'max_freq.csv') 
        max_freq_climax_csv = os.path.join(config.video_path,'max_climax_freq.csv') 

        text_file = open(os.path.join(config.video_path,'process.txt'), "w")

        main(config.video_path)
        

import librosa
import os
import numpy as np
import pyworld as pw
import python_speech_features
import scipy.io.wavfile as wav
from nnmnkwii.preprocessing import interp1d
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing import cpu_count
from tqdm import tqdm
import argparse
from natsort import natsorted

hop_size = 160 # 10ms
frame_length = 400 # 25ms
frame_period = 10

def proc_wav(wav_path, out_dir, index, spkid, sr=16000):
    fs, signal = wav.read(wav_path)
    frame_count = int(len(signal)/hop_size)
    pad_len = (frame_count-1) * hop_size + frame_length
    padded = np.pad(signal, (0, pad_len-len(signal)), mode="constant", constant_values=0)
    mfcc = python_speech_features.mfcc(padded, fs, winlen=0.025, winstep=0.01, nfilt=40, numcep=13) # log-energy, mfcc[1:13]
    out = padded[:frame_count*hop_size]
    assert len(out) % mfcc.shape[0] == 0
    padded = padded.astype(np.float64)
    f0, timeaxis = pw.harvest(padded, fs, frame_period=frame_period) 
    f0 = f0[:frame_count]
    vuv = np.zeros(len(f0))
    vuv[f0>0] = 1
    logf0 = np.zeros(len(f0))
    logf0[f0>0] = np.log(f0[f0>0])
    continuous_lf0 = interp1d(logf0, kind="slinear")
    print (continuous_lf0.shape, vuv.shape, out.shape, mfcc.shape)

def build_from_path(in_dir, out_dir, spkid, num_workers=1, tqdm=lambda x: x):
    os.makedirs(out_dir, exist_ok=True)
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    index = 1
    file_names = natsorted(
        [os.path.join(in_dir, file_name) for file_name in os.listdir(in_dir)]
    )
    for wav_path in file_names:
        futures.append(executor.submit(
            partial(proc_wav, wav_path, out_dir, index, spkid)))
        index += 1
    return [future.result() for future in tqdm(futures)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='preprocess for erlangshen')
    parser.add_argument('in_dir', type=str, help='input dir')
    parser.add_argument('out_dir', type=str, help='output_dir')
    parser.add_argument('spkid', type=str, help='spk name')
    args = parser.parse_args()

    number_workers = int(cpu_count()/2)
    print (number_workers)
    build_from_path(args.in_dir, args.out_dir, args.spkid, number_workers, tqdm=tqdm)

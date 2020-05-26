import pyaudio
import wave
import matplotlib.pyplot as plt
import numpy as np
import wave
import sys
import time
from random import random as rand
from pylsl import StreamInfo, StreamOutlet
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from scipy import signal
from scipy.signal import hilbert
import matplotlib
from scipy.signal import freqs

def butter_lowpass(cutOff, fs, order=5):
    nyq = 0.5 * fs
    normalCutoff = cutOff / nyq
    b, a = butter(order, normalCutoff, btype='low', analog = True)
    return b, a

def butter_lowpass_filter(data, cutOff, fs, order=4):
    b, a = butter_lowpass(cutOff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

cutOff = 23.1 #cutoff frequency in rad/s
fs_lowpass = 188.495559 #sampling frequency in rad/s
order = 5 #order of filter

volume = 0.5  # range [0.0, 1.0]
fs = 44100 # sampling rate, Hz, must be integer
duration = 5  # in seconds, may be float
f_sin = 1000  # sine frequency, Hz, may be float

#print sticker_data.ps1_dxdt2

info = StreamInfo('BioSemi', 'EEG', 1, 4096, 'float32', 'myuid34234')

# next make an outlet
outlet = StreamOutlet(info)

p = pyaudio.PyAudio()
info = p.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')
for i in range(0, numdevices):
        if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))

calibrate_times = 5
avg_15 = 0
print("now sending data...")
while True:
    # make a new random 8-channel sample; this is converted into a
    # pylsl.vectorf (the data type that is expected by push_sample)
    # now send it and wait for a bit

    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    RECORD_SECONDS = 5
    WAVE_OUTPUT_FILENAME = "output.wav"

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                    input_device_index=1)



    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)



    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()


    input_data = read(WAVE_OUTPUT_FILENAME)
    # audio = [np.average(input_data[1])]
    original_audio = input_data[1]
    # sine = (np.sin(2 * np.pi * np.arange(fs * duration) * f_sin / fs)).astype(np.float32)
    # audio = np.array([x*y for x, y in zip(original_audio, sine)])
    # audio = butter_bandpass_filter(original_audio, 955, 1045, fs, order)
    analytic_signal = hilbert(original_audio)
    amplitude_envelope = np.abs(analytic_signal)
    amplitude_envelope = amplitude_envelope - amplitude_envelope.mean()

    ps = np.abs(np.fft.fft(amplitude_envelope)) ** 2

    time_step = 1 / 44100
    freqs = np.fft.fftfreq(amplitude_envelope.size, time_step)
    idx = np.argsort(freqs)
    #
    #
    # f, t, Sxx = signal.spectrogram(audio, fs)
    plt.subplot(3, 1, 1)
    # plt.pcolormesh(t, f, Sxx)
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    axes = plt.gca()
    axes.set_ylim([-40000, 40000])
    plt.plot(original_audio)

    plt.subplot(3, 1, 2)
    axes = plt.gca()
    axes.set_ylim([-12000, 12000])
    plt.plot(amplitude_envelope)

    plt.subplot(3, 1, 3)
    axes = plt.gca()
    axes.set_xlim([10, 20])
    axes.set_ylim([0, 4e16])

    plt.plot(freqs[idx], ps[idx])
    plt.grid(True)
    plt.show()

    print(f'the 15 value is {np.average(ps[np.where((freqs >= 14) & (freqs <=16))])/1e16}')

    if calibrate_times > 0:
        avg_15 += np.average(ps[np.where((freqs >= 14) & (freqs <=16))])
    elif calibrate_times == 0:
        avg_15 /= 5
        print(f'the threshold is {avg_15}')

    if calibrate_times < 0:
        if np.average(ps[np.where((freqs >= 14) & (freqs <=16))]) >= avg_15 * 1.3:
            print('I recognized the frequency')
        else:
            print('I did"t recognize the frequency')

    calibrate_times -= 1
    # print(f'sent {audio}')
    # outlet.push_chunk(audio)
    time.sleep(0.01)



    # # read audio samples
    #
    # # plot the first 1024 samples
    # plt.plot(audio[0:1024])
    # # label the axes
    # plt.ylabel("Amplitude")
    # plt.xlabel("Time")
    # # set the title
    # plt.title("Sample Wav")
    # # display the plot
    # plt.show()


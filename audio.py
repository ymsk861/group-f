# -*- coding:utf-8 -*-

import pyaudio

CHUNK=1024*2
RATE=44100
p=pyaudio.PyAudio()

stream=p.open( format = pyaudio.paInt16,
                channels = 1,
                rate = RATE,
                frames_per_buffer = CHUNK,
                input = True,
                output = True) # inputとoutputを同時にTrueにする


def audio_trans(input):
    # なんかしらの処理
    print("yeah")
    print(type(input))
    return input

while stream.is_active():
    input = stream.read(CHUNK)
    input = audio_trans(input)
    output = stream.write(input)

stream.stop_stream()
stream.close()
p.terminate()

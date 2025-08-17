import queue, sys, signal
import sounddevice as sd
import soundfile as sf

RATE = 48000      # sample rate (Hz)
CHANNELS = 1      # 1=mono, 2=stereo
BLOCKSIZE = 1024
DEVICE = None     # set to an input device index/name if needed
OUT = "mic_recording.wav"

q = queue.Queue()
print(sd.query_devices()) # check which sound device it is 
def callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(indata.copy())

signal.signal(signal.SIGINT, signal.SIG_DFL)  # clean Ctrl+C on Windows

with sf.SoundFile(OUT, mode='w', samplerate=RATE, channels=CHANNELS, subtype='PCM_16') as wav, \
     sd.InputStream(device=DEVICE, channels=CHANNELS, samplerate=RATE, blocksize=BLOCKSIZE, callback=callback):
    print(f"Recording… Press Ctrl+C to stop. Saving to {OUT}")
    while True:
        wav.write(q.get())
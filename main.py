import numpy as np
import sounddevice as sd
import soundfile as sf
from collections import deque
from scipy.signal import butter, lfilter

# ---------- config ----------
REFERENCE_WAV = "reference.wav"
THRESHOLD = 0.3      # lowered threshold for better detection; tune up/down
CHUNK_SECONDS = 1.0     # record 1-second chunks
BANDPASS = (500, 6000)  # Hz; claps are broadband but this removes rumble/hiss
PRINT_ALL = True
# ----------------------------

def to_mono(x):
    return x if x.ndim == 1 else np.mean(x, axis=1)

def bandpass(y, sr, lo_hz, hi_hz, order=4):
    ny = 0.5 * sr
    lo = lo_hz / ny
    hi = hi_hz / ny
    b, a = butter(order, [lo, hi], btype="band")
    return lfilter(b, a, y).astype(np.float32)

def zmean_l2norm(y, eps=1e-12):
    y = y - np.mean(y)
    n = np.linalg.norm(y) + eps
    return (y / n).astype(np.float32)

def peak_xcorr_norm(buffer_vec, ref_vec):
    """
    Fast-ish normalized cross-correlation peak:
    We z-normalize both signals once (approximate NCC).
    Then use valid-mode correlation across all lags and return the peak.
    """
    x = zmean_l2norm(buffer_vec)
    r = zmean_l2norm(ref_vec)
    # valid correlation: slide r across x
    # np.correlate returns sum(x[i:i+len(r)] * r[::-1])
    # Using reversed ref is equivalent for correlation; peak value in [-1,1]
    peak = np.max(np.correlate(x, r, mode="valid"))
    return float(peak)

# 1) Load & prep reference
ref_wav, ref_sr = sf.read(REFERENCE_WAV, always_2d=False)
ref_wav = to_mono(np.asarray(ref_wav, dtype=np.float32))

# Band-pass the reference (helps robustness across rooms/mics)
ref_wav = bandpass(ref_wav, ref_sr, BANDPASS[0], BANDPASS[1])

# Keep a slightly shorter reference (optional): trim leading/trailing silence if needed
# (For claps it's usually already short. You can slice if your file has silence.)
ref_len = len(ref_wav)

# 2) Buffer for 1-second chunks  
sr = ref_sr
chunk_samples = int(CHUNK_SECONDS * sr)  # 1-second worth of samples
# Use maxlen to automatically manage buffer size - keeps most recent audio
buffer = deque(maxlen=max(chunk_samples, ref_len * 2))  # Ensure buffer can hold enough for correlation
chunk_ready = False

def audio_callback(indata, frames, time, status):
    global chunk_ready
    if status:
        print("Audio status:", status)
    mono = to_mono(indata.copy()).astype(np.float32)
    print(f"DEBUG: Received {len(mono)} samples, max level: {np.max(np.abs(mono)):.3f}")
    
    # Band-pass live audio the same way we did the reference
    mono = bandpass(mono, sr, BANDPASS[0], BANDPASS[1])
    buffer.extend(mono)
    
    # Mark chunk as ready when buffer has enough samples for processing
    if len(buffer) >= chunk_samples:
        chunk_ready = True
        print(f"DEBUG: Chunk ready! Buffer filled to {len(buffer)} samples")

print(
    f"Listening {sr} Hz | ref {ref_len/sr:.3f}s | "
    f"chunk {CHUNK_SECONDS:.1f}s | thr {THRESHOLD}"
)

# 4) Stream & check
try:
    with sd.InputStream(channels=1, samplerate=sr, callback=audio_callback, blocksize=chunk_samples//10):
        while True:
            if chunk_ready:
                # Process the chunk
                buf_np = np.array(buffer, dtype=np.float32)
                audio_level = np.max(np.abs(buf_np))
                print(f"DEBUG: Buffer size: {len(buf_np)}, Audio level: {audio_level:.3f}")
                
                # Check if buffer is long enough for correlation
                if len(buf_np) >= ref_len:
                    peak = peak_xcorr_norm(buf_np, ref_wav)
                    
                    if peak >= THRESHOLD:
                        print(f"🎯 CLAP DETECTED! Similarity: {peak:.3f} (threshold: {THRESHOLD})")
                    elif PRINT_ALL:
                        print(f"🔍 Similarity: {peak:.3f} (threshold: {THRESHOLD}) - No match")
                else:
                    print(f"DEBUG: Buffer too short: {len(buf_np)} < {ref_len}")
                
                # Reset chunk ready flag but keep buffer data flowing
                chunk_ready = False
                # Don't clear buffer - let deque maxlen handle overflow automatically

            sd.sleep(100)  # Small sleep to prevent busy waiting
except KeyboardInterrupt:
    print("\nStopped.")
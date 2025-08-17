import numpy as np
import sounddevice as sd
import soundfile as sf
from collections import deque
from scipy.signal import butter, lfilter
from scipy.fft import fft, fftfreq
import threading
import time
import subprocess
import os
import webbrowser

# ---------- config ----------
REFERENCE_WAV = "reference.wav"
WAKEUP_WAV = "wakeup.wav"    # Audio file to play when clap is detected
WAKEUP_VOLUME = 0.3          # Volume level (0.0 to 1.0)
LEAGUE_SHORTCUT_PATH = r"C:\Riot Games\Riot Client\RiotClientServices.exe"  # League of Legends desktop shortcut
THRESHOLD = 0.3      # Much lower onset detection threshold
CHUNK_SECONDS = 0.3  # Even shorter chunks for better responsiveness  
SPECTRAL_THRESHOLD = 0.7  # Higher spectral similarity threshold to reduce false positives
ENERGY_THRESHOLD = 0.01    # Higher minimum energy for detection  
ONSET_THRESHOLD = 1.8      # Higher onset detection sensitivity
# ----------------------------

def to_mono(x):
    return x if x.ndim == 1 else np.mean(x, axis=1)

# Global variable to track playback state
is_playing = False
playback_lock = threading.Lock()

def play_wakeup_sound():
    """Play the wakeup sound in a separate thread"""
    global is_playing
    
    with playback_lock:
        if is_playing:
            print("🔇 Wakeup sound already playing, skipping...")
            return
        is_playing = True
    
    try:
        print(f"🔊 Playing {WAKEUP_WAV} at {WAKEUP_VOLUME*100:.0f}% volume...")
        wakeup_data, wakeup_sr = sf.read(WAKEUP_WAV, always_2d=False)
        wakeup_data = to_mono(np.asarray(wakeup_data, dtype=np.float32))
        
        # Reduce volume
        wakeup_data = wakeup_data * WAKEUP_VOLUME
        
        # Play the audio
        sd.play(wakeup_data, wakeup_sr)
        sd.wait()  # Wait until playback is finished
        print("✅ Wakeup sound finished playing")
        
    except Exception as e:
        print(f"❌ Error playing wakeup sound: {e}")
    finally:
        with playback_lock:
            is_playing = False

def launch_league_of_legends():
    """Launch League of Legends and open YouTube search for Nujabes"""
    # Open YouTube search for Nujabes
    try:
        webbrowser.open("https://www.youtube.com/results?search_query=Nujabes")
        print("Opening YouTube search for Nujabes...")
    except Exception as e:
        print(f"❌ Error opening YouTube: {e}")
    
    # Check if the file exists
    if os.path.exists(LEAGUE_SHORTCUT_PATH):
        try:
            subprocess.Popen([LEAGUE_SHORTCUT_PATH], shell=True)  # Opens League of Legends
            print("Launching League of Legends...")
        except PermissionError:
            print("❌ Permission denied. Try running this script as Administrator.")
        except Exception as e:
            print(f"❌ Error launching League of Legends: {e}")
    else:
        print("League of Legends not found at:", LEAGUE_SHORTCUT_PATH)

def trigger_wakeup():
    """Trigger wakeup sound and launch League of Legends in background thread"""
    global is_playing
    
    # Check if already playing
    with playback_lock:
        if is_playing:
            print("⏩ Clap detected but wakeup sound already playing")
            return
    
    # Use thread to avoid blocking audio detection
    thread = threading.Thread(target=play_wakeup_sound)
    thread.daemon = True
    thread.start()
    
    # Launch League of Legends (non-blocking)
    game_thread = threading.Thread(target=launch_league_of_legends)
    game_thread.daemon = True
    game_thread.start()

def get_spectral_features(signal, sr):
    """Extract spectral features from audio signal"""
    # Apply window to reduce edge effects
    windowed = signal * np.hanning(len(signal))
    
    # FFT
    fft_vals = np.abs(fft(windowed))
    freqs = fftfreq(len(signal), 1/sr)
    
    # Only positive frequencies
    pos_mask = freqs >= 0
    fft_vals = fft_vals[pos_mask]
    freqs = freqs[pos_mask]
    
    # Focus on clap frequency range (500Hz - 8kHz)
    freq_mask = (freqs >= 500) & (freqs <= 8000)
    clap_spectrum = fft_vals[freq_mask]
    
    # Normalize
    if np.max(clap_spectrum) > 0:
        clap_spectrum = clap_spectrum / np.max(clap_spectrum)
    
    return clap_spectrum

def detect_onset(signal, sr, threshold=1.2):
    """Detect sharp onset using multiple methods"""
    
    # Method 1: Simple energy increase detection
    frame_size = len(signal) // 8  # Smaller frames for better sensitivity
    energy_ratios = []
    
    for i in range(0, len(signal) - frame_size, frame_size // 4):
        frame1 = signal[i:i + frame_size]
        frame2 = signal[i + frame_size//2:i + frame_size//2 + frame_size]
        
        if len(frame2) < frame_size:
            break
            
        energy1 = np.sum(frame1**2)
        energy2 = np.sum(frame2**2)
        
        if energy1 > 1e-8:  # Avoid division by zero
            ratio = energy2 / energy1
            energy_ratios.append(ratio)
    
    # Method 2: High-frequency energy detection (claps have high freq content)
    fft_vals = np.abs(fft(signal * np.hanning(len(signal))))
    freqs = fftfreq(len(signal), 1/sr)
    
    # High frequency energy (2kHz+)
    high_freq_mask = (freqs >= 2000) & (freqs <= sr/2)
    high_freq_energy = np.sum(fft_vals[high_freq_mask]**2)
    
    # Total energy
    total_energy = np.sum(fft_vals**2)
    high_freq_ratio = high_freq_energy / (total_energy + 1e-8)
    
    # Combine both methods
    max_energy_ratio = np.max(energy_ratios) if energy_ratios else 1.0
    onset_strength = max_energy_ratio * (1 + high_freq_ratio * 5)  # Weight high freq content
    
    return onset_strength

def spectral_similarity(spec1, spec2):
    """Calculate cosine similarity between spectra"""
    if len(spec1) != len(spec2):
        min_len = min(len(spec1), len(spec2))
        spec1 = spec1[:min_len]
        spec2 = spec2[:min_len]
    
    # Cosine similarity
    dot_product = np.dot(spec1, spec2)
    norm1 = np.linalg.norm(spec1)
    norm2 = np.linalg.norm(spec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    similarity = dot_product / (norm1 * norm2)
    return similarity

# Load reference
print(f"Loading reference file: {REFERENCE_WAV}")
ref_wav, ref_sr = sf.read(REFERENCE_WAV, always_2d=False)
ref_wav = to_mono(np.asarray(ref_wav, dtype=np.float32))
print(f"Reference: {len(ref_wav)} samples, {len(ref_wav)/ref_sr:.3f}s, sr={ref_sr}Hz")

# Check if wakeup file exists
try:
    wakeup_test, wakeup_test_sr = sf.read(WAKEUP_WAV, always_2d=False)
    print(f"✅ Wakeup file loaded: {WAKEUP_WAV} ({len(wakeup_test)} samples, {len(wakeup_test)/wakeup_test_sr:.3f}s)")
except Exception as e:
    print(f"❌ Error loading wakeup file {WAKEUP_WAV}: {e}")
    print("   Make sure wakeup.wav exists in the current directory!")

# Check if League of Legends shortcut exists
if os.path.exists(LEAGUE_SHORTCUT_PATH):
    print(f"✅ League of Legends shortcut found: {LEAGUE_SHORTCUT_PATH}")
else:
    print(f"⚠️  League of Legends shortcut not found: {LEAGUE_SHORTCUT_PATH}")
    
    # List desktop files to help find the correct name
    desktop_path = r"C:\Users\House Computer\Desktop"
    if os.path.exists(desktop_path):
        print("📂 Files on desktop:")
        try:
            for file in os.listdir(desktop_path):
                if "league" in file.lower() or file.endswith(".lnk"):
                    print(f"   - {file}")
        except:
            print("   Unable to list desktop files")
    print("   Will attempt to launch anyway (may fail)")

# Extract reference features
ref_spectrum = get_spectral_features(ref_wav, ref_sr)
ref_onset = detect_onset(ref_wav, ref_sr)
print(f"Reference onset strength: {ref_onset:.3f}")
print(f"Reference spectrum shape: {ref_spectrum.shape}")

# Audio processing
sr = ref_sr
chunk_samples = int(CHUNK_SECONDS * sr)
buffer = deque(maxlen=chunk_samples)
chunk_ready = False

# Adaptive thresholding
background_energy = 0.001  # Initialize with low value
energy_history = deque(maxlen=20)  # Keep recent energy levels

def audio_callback(indata, frames, time, status):
    global chunk_ready
    if status:
        print("Audio status:", status)
    
    mono = to_mono(indata.copy()).astype(np.float32)
    buffer.extend(mono)
    
    if len(buffer) >= chunk_samples:
        chunk_ready = True

print(f"Listening {sr}Hz | chunk {CHUNK_SECONDS:.1f}s | onset_thr {ONSET_THRESHOLD}")
print(f"Spectral threshold: {SPECTRAL_THRESHOLD}, Energy threshold: {ENERGY_THRESHOLD}")

# Stream and detect
try:
    with sd.InputStream(channels=1, samplerate=sr, callback=audio_callback, blocksize=chunk_samples//10):
        while True:
            if chunk_ready:
                buf_np = np.array(buffer, dtype=np.float32)
                energy = np.sqrt(np.mean(buf_np**2))
                
                # Update background energy estimate
                energy_history.append(energy)
                if len(energy_history) > 5:
                    background_energy = np.median(sorted(energy_history)[:10])  # Use lower values for background
                
                # Adaptive energy threshold - more strict
                adaptive_threshold = max(background_energy * 5, ENERGY_THRESHOLD)
                
                print(f"Energy: {energy:.6f}, Background: {background_energy:.6f}, Threshold: {adaptive_threshold:.6f}")
                
                if energy > adaptive_threshold:
                    # Detect onset
                    onset_strength = detect_onset(buf_np, sr, ONSET_THRESHOLD)
                    print(f"Onset: {onset_strength:.3f} (threshold: {ONSET_THRESHOLD})")
                    
                    # Strict requirements - BOTH onset AND spectral match required
                    if onset_strength > ONSET_THRESHOLD:
                        # Check spectral similarity
                        current_spectrum = get_spectral_features(buf_np, sr)
                        similarity = spectral_similarity(current_spectrum, ref_spectrum)
                        
                        print(f"🎵 ONSET DETECTED: Onset={onset_strength:.3f}, Similarity={similarity:.6f}")
                        
                        # Require BOTH strong onset AND high spectral similarity
                        if similarity > SPECTRAL_THRESHOLD and onset_strength > ONSET_THRESHOLD:
                            print(f"🎯 CLAP DETECTED! Onset: {onset_strength:.3f}, Similarity: {similarity:.6f}")
                            trigger_wakeup()  # Play the wakeup sound!
                        else:
                            print(f"❌ Not a matching clap. Onset: {onset_strength:.3f}, Similarity: {similarity:.6f}")
                    else:
                        print(f"🔍 Onset too weak: {onset_strength:.3f} < {ONSET_THRESHOLD}")
                else:
                    print(f"🔇 Too quiet: {energy:.6f} < {adaptive_threshold:.6f}")
                
                chunk_ready = False
                
            sd.sleep(50)
            
except KeyboardInterrupt:
    print("\nStopped.")
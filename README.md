# Wake Up Daddy's Home - Clap Detection System

A Python script that listens for clap sounds and automatically launches League of Legends while opening a YouTube search for Nujabes music.

## Features

- **Real-time clap detection** using audio analysis
- **Automatic League of Legends launch** when clap is detected
- **YouTube music integration** - opens Nujabes search results
- **Wakeup sound playback** with volume control
- **Advanced audio processing** with spectral analysis and onset detection

## Requirements

```
numpy
sounddevice
soundfile
scipy
```

## Setup

1. Install dependencies:
   ```bash
   pip install numpy sounddevice soundfile scipy
   ```

2. Place required audio files in the project directory:
   - `reference.wav` - Reference clap sound for detection
   - `wakeup.wav` - Sound to play when clap is detected

3. Update the League of Legends path in `main.py` if needed:
   ```python
   LEAGUE_SHORTCUT_PATH = r"C:\Riot Games\Riot Client\RiotClientServices.exe"
   ```

## Usage

Run the main script:
```bash
python main.py
```

The script will:
1. Load reference audio files
2. Start listening for clap sounds
3. When a clap is detected:
   - Play the wakeup sound
   - Open YouTube search for Nujabes
   - Launch League of Legends

## Configuration

Adjust detection sensitivity in `main.py`:

- `THRESHOLD` - Clap detection sensitivity
- `SPECTRAL_THRESHOLD` - Spectral similarity threshold
- `ENERGY_THRESHOLD` - Minimum energy for detection
- `ONSET_THRESHOLD` - Onset detection sensitivity
- `WAKEUP_VOLUME` - Volume level for wakeup sound (0.0 to 1.0)

## Files

- `main.py` - Main clap detection and response system
- `open_league.py` - Simple League of Legends launcher
- `reference.wav` - Reference clap audio
- `wakeup.wav` - Wakeup sound file

## How It Works

1. **Audio Processing**: Captures real-time audio input
2. **Spectral Analysis**: Extracts frequency features from audio
3. **Onset Detection**: Identifies sharp sound onsets characteristic of claps
4. **Pattern Matching**: Compares detected sounds to reference clap
5. **Action Trigger**: Launches applications and plays sounds when match found

Press `Ctrl+C` to stop the detection system.
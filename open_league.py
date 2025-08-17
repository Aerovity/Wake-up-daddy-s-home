import os
import subprocess

# Path to League of Legends client
league_path = r"C:\Riot Games\League of Legends\LeagueClient.exe"

# Check if the file exists
if os.path.exists(league_path):
    subprocess.Popen(league_path)  # Opens League of Legends
    print("Launching League of Legends...")
else:
    print("League of Legends not found at:", league_path)
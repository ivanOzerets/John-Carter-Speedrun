import yt_dlp
import whisper
import os
import boto3

# download mp3s
file_directory = "data/"
file_name = "urls.md"

file_path = os.path.join(file_directory, file_name)

if os.path.exists(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        url_file = file.read()
else:
    print(f"error: the file at {file_path} does not exist.")

ydl_opts = {
    'format': 'm4a/bestaudio/best',
    # ℹ️ See help(yt_dlp.postprocessor) for a list of available Postprocessors and their arguments
    'postprocessors': [{  # Extract audio using ffmpeg
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'm4a',
    }]
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:

    _ = ydl.download(URLS)

# run mp3s through whisper

# chunk text files i guess

# feed into bedrock for embeddings

# create or return some sort of container full of the embeddings
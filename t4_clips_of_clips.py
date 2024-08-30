from pydub import AudioSegment
import pandas as pd
import whisper
import json

audio_path = "kit_audio_extract_2min.mp3"
rttm_path = "kit_audio_extract_2min_hf.rttm"

# Load the Whisper model
model = whisper.load_model("base.en")

# Load the audio file
audio = AudioSegment.from_file(audio_path)

# Read the RTTM file
rttm_df = pd.read_csv(rttm_path, delim_whitespace=True, header=None,
                      names=["Type", "File ID", "Channel ID", "Start Time", "Duration", "Orthography Field", 
                             "Speaker Type", "Speaker Name", "Confidence Score", "Signal Lookahead Time"])

# Initialize a list to store the results
transcriptions = []

# Iterate over each row in the RTTM DataFrame
for index, row in rttm_df.iterrows():
    start_time = row["Start Time"] * 1000  # Convert to milliseconds
    duration = row["Duration"] * 1000  # Convert to milliseconds
    end_time = start_time + duration

    # Extract the segment from the audio
    segment = audio[start_time:end_time]

    # Export the segment to a temporary file
    segment.export("temp_segment.wav", format="wav")

    # Transcribe the audio segment using Whisper
    result = model.transcribe("temp_segment.wav")

    # Append the transcription and speaker name to the list
    transcriptions.append({
        "speaker_name": row["Speaker Name"],
        "transcription": result["text"]
    })

# Save the result to a JSON file
with open("trans_clips.json", "w") as f:
    json.dump(transcriptions, f, indent=4)

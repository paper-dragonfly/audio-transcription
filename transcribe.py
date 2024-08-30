import json
from pyannote.audio import Pipeline
import whisper
import yaml
import pdb

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

audio_file = "kit_clip_45sec.mp3"
access_token = config["pyannote_hf_access_token"]

# pdb.set_trace()
# Load the Pyannote diarization pipeline
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1", 
    use_auth_token=access_token)

# send to GPU
import torch
pipeline.to(torch.device("cuda"))


# Load the Whisper model
whisper_model = whisper.load_model("base")

# Perform speaker diarization
diarization = pipeline(audio_file)

# Transcribe the audio using Whisper
audio = whisper.load_audio(audio_file)
audio = whisper.pad_or_trim(audio)
mel = whisper.log_mel_spectrogram(audio).to(whisper_model.device)
options = whisper.DecodingOptions(language="en", fp16=False)
transcription_result = whisper_model.transcribe(audio_file, options=options)

# Combine diarization and transcription results
transcription_with_speakers = []
for segment in diarization.get_timeline().support():
    start, end = segment.start, segment.end
    speaker = diarization.get_labels(segment)[0]
    text_segment = next((item['text'] for item in transcription_result['segments'] if item['start'] >= start and item['end'] <= end), "")
    transcription_with_speakers.append({
        "speaker": speaker,
        "start": start,
        "end": end,
        "text": text_segment
    })

# Save the result to a JSON file
with open("transcription.json", "w") as f:
    json.dump(transcription_with_speakers, f, indent=4)


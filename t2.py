import json
from pyannote.audio import Pipeline
import whisper
import yaml
import torch

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

audio_file = "kit_clip_45sec.mp3"
access_token = config["pyannote_hf_access_token"]

# Load the Pyannote diarization pipeline
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1", 
    use_auth_token=access_token)

# Send to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipeline.to(device)

# Load the Whisper model
whisper_model = whisper.load_model("base.en", device=device)

# Perform speaker diarization
diarization = pipeline(audio_file, num_speakers=3)

# Print and save the diarization object for inspection
with open("diarization.json", "w") as f:
    f.write(str(diarization))

# Transcribe the audio using Whisper
transcription_result = whisper_model.transcribe(audio_file, language="en", fp16=False)

# Print and save the transcription result for inspection
with open("transcription_result.json", "w") as f:
    json.dump(transcription_result, f, indent=4)

# Combine diarization and transcription results
transcription_with_speakers = []
for segment, _, speaker in diarization.itertracks(yield_label=True):
    start, end = segment.start, segment.end
    text_segment = next((item['text'] for item in transcription_result['segments'] 
                         if item['start'] >= start and item['end'] <= end), "")
    transcription_with_speakers.append({
        "speaker": speaker,
        "start": start,
        "end": end,
        "text": text_segment
    })
    

# Save the result to a JSON file
with open("transcription.json", "w") as f:
    json.dump(transcription_with_speakers, f, indent=4)

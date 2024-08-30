import json

with open("transcription_result.json", 'r') as f:
    whisper_trans = json.load(f)


for i in range(len(whisper_trans["segments"])):
    seg = whisper_trans["segments"][i]
    print(f"""ID: {seg["id"]} | {seg["start"]} | {seg["end"]}""")
import whisper

model = whisper.load_model("base.en")
result = model.transcribe("kit_clip_45sec.mp3")
print(result["text"])
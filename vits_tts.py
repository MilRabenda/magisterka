from TTS.api import TTS

tts = TTS("tts_models/pl/mai_female/vits", progress_bar=False, gpu=False)

text = "Cześć, to jest test mowy po polsku. Hello world!"
tts.tts_to_file(text=text, file_path="polish.wav")

print("Audio saved to polish.wav")

from TTS.api import TTS
import soundfile as sf

tts = TTS("tts_models/pl/mai_female/vits")

text = "Cześć, to jest test mowy po polsku!"
audio = tts.tts(text=text)

sf.write("xtts_polish.wav", audio, tts.synthesizer.output_sample_rate)
print("Saved to xtts_polish.wav")

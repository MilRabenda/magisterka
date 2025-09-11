from transformers import pipeline

pipe = pipeline("text-to-speech", model="salihfurkaan/VoxPolska-Auralis")
output = pipe("Cześć, jestem modelem mówiącym po polsku.")

import soundfile as sf
sf.write("output.wav", output["audio"], output["sampling_rate"])
print("Audio saved!")

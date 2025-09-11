from transformers import VitsModel, AutoTokenizer
import torch
import numpy as np
from scipy.io.wavfile import write

model = VitsModel.from_pretrained("facebook/mms-tts-pol")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-pol")

text = "Cześć, to jest test mowy po polsku. Hello world!"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    output = model(**inputs).waveform

waveform = output[0].cpu().numpy()

waveform_int16 = np.int16(waveform / np.max(np.abs(waveform)) * 32767)

write("polish_meta.wav", model.config.sampling_rate, waveform_int16)

print("Saved to polish_meta.wav")

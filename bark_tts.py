import torch
import numpy as np
import soundfile as sf
from bark import generate_audio, generation

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cpu":
    print("No GPU being used. Inference will be slow!")

_original_torch_load = torch.load

def torch_load_safe(*args, **kwargs):
    kwargs['weights_only'] = False
    try:
        import numpy.core.multiarray as multiarray
        with torch.serialization.safe_globals([multiarray.scalar, np.dtype, np.generic]):
            return _original_torch_load(*args, **kwargs)
    except (ImportError, AttributeError):
        with torch.serialization.safe_globals([np.dtype, np.generic]):
            return _original_torch_load(*args, **kwargs)

torch.load = torch_load_safe

generation.preload_models()

torch.load = _original_torch_load

audio_array = generate_audio("Cześć, to jest test mowy po polsku!")
audio_array = np.clip(audio_array, -1.0, 1.0)

sf.write("bark_polish.wav", audio_array, 24000, subtype='FLOAT')
print("Saved to bark_polish.wav")

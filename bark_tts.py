import torch
import numpy as np
import soundfile as sf
from bark import generate_audio, generation
import time
import os

# --- GPU Setup ---
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
print(f"Using device: {device}")
if device == "cpu":
    print("WARNING: No GPU detected. Inference will be very slow!")

# ---Safe Load Hack ---
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

print("Preloading Bark models...")
generation.preload_models(
    text_use_gpu=(device == "cuda"),
    text_use_small=False,
    coarse_use_gpu=(device == "cuda"),
    coarse_use_small=False,
    fine_use_gpu=(device == "cuda"),
    fine_use_small=False
)

torch.load = _original_torch_load

# --- Scenarios ---

scenarios = [
    {
        "desc": "Poważne / Podniosłe",
        "speaker": "v2/pl_speaker_0",  # Usually a deep, steady male voice
        "text": "Litwo! Ojczyzno moja! ty jesteś jak zdrowie. Ile cię trzeba cenić, ten tylko się dowie, Kto cię stracił. Dziś piękność twą w całej ozdobie, Widzę i opisuję, bo tęsknię po tobie."
    },
    {
        "desc": "Radosne / Entuzjastyczne",
        "speaker": "v2/pl_speaker_9",  # Usually a higher pitched/female voice
        "text": "[laughter] Marian! Tu jest jakby luksusowo! Widzisz to słońce? Czuję, że dzisiaj wydarzy się coś absolutnie wspaniałego. Aż chce się żyć, tańczyć i śpiewać!"
    },
    {
        "desc": "Informacyjne / Neutralne",
        "speaker": "v2/pl_speaker_5",  # Standard, slightly flatter tone
        "text": "Pociąg Intercity relacji Warszawa Centralna – Gdynia Główna wjedzie na tor drugi przy peronie trzecim. Prosimy o zachowanie ostrożności i odsunięcie się od krawędzi peronu."
    },
    {
        "desc": "Smutne / Melancholijne",
        "speaker": "v2/pl_speaker_3",  # Softer tone
        "text": "[sigh] Umrzeć – tego nie robi się kotu. Bo co ma począć kot w pustym mieszkaniu? Drapać się na ściany? Ocierać o meble? Nic niby tu nie zmienione, a jednak wszystko się pozamieniało."
    },
    {
        "desc": "Tajemnicze / Napięcie",
        "speaker": "v2/pl_speaker_8",  # Often lower quality/gritty or whispery
        "text": "[whisper] Słyszysz to? To nie był wiatr. Kroki na schodach stają się coraz głośniejsze. Ktoś tu jest... stoi tuż za drzwiami i czeka, aż zgaśnie ostatnia świeca."
    }
]

execution_times = []
#output_dir = "bark_outputs"
output_dir = "bark_outputs_cpu"
os.makedirs(output_dir, exist_ok=True)

print(f"\nStarting Batch Generation for {len(scenarios)} scenarios...\n" + "-" * 50)

for i, item in enumerate(scenarios):
    print(f"Generating [{i + 1}/{len(scenarios)}]: {item['desc']}")
    print(f" -> Speaker: {item['speaker']}")

    start_time = time.time()

    audio_array = generate_audio(item["text"], history_prompt=item['speaker'])

    end_time = time.time()
    duration = end_time - start_time
    execution_times.append(duration)

    audio_array = np.clip(audio_array, -1.0, 1.0)

    safe_desc = item['desc'].split("/")[0].strip().replace(" ", "_")
    filename = f"{output_dir}/{i + 1}_{safe_desc}.wav"

    sf.write(filename, audio_array, 24000, subtype='FLOAT')

    print(f" -> Finished in {duration:.2f}s. Saved: {filename}")
    print("-" * 20)

# --- Final Summary ---
print("\n" + "=" * 60)
print("PERFORMANCE SUMMARY")
print("=" * 60)
print(f"{'#':<4} {'Time (s)':<10} {'Emotion/Type':<25} {'Speaker'}")
print("-" * 60)

for i, (duration, item) in enumerate(zip(execution_times, scenarios)):
    print(f"{i + 1:<4} {duration:<10.2f} {item['desc']:<25} {item['speaker']}")

print("-" * 60)
total_time = sum(execution_times)
print(f"Total processing time: {total_time:.2f} seconds")
print(f"Average time per clip: {total_time / len(scenarios):.2f} seconds")
print("=" * 60)
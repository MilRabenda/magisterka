import time
import os
import torch
import sys
from TTS.api import TTS

try:
    _original_torch_load = torch.load


    def safe_torch_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)


    torch.load = safe_torch_load
except AttributeError:
    pass

# --- GPU Setup ---
use_gpu = torch.cuda.is_available()

#device_name = "cuda" if use_gpu else "cpu"
device_name = "cpu"
print(f"Using device: {device_name}")

# --- CONFIGURATION ---
AUDIO_SAMPLES_DIR = r"C:\Users\milen\PycharmProjects\PythonProject\tts\audio_samples"
OUTPUT_DIR = "xtts_outputs_cpu"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- SCENARIOS ---
scenarios = [
    {
        "desc": "Poważne / Podniosłe",
        "file_name": "shrek-poważne.mp3",
        "text": "Litwo! Ojczyzno moja! ty jesteś jak zdrowie. Ile cię trzeba cenić, ten tylko się dowie, Kto cię stracił. Dziś piękność twą w całej ozdobie, Widzę i opisuję, bo tęsknię po tobie."
    },
    {
        "desc": "Radosne / Entuzjastyczne",
        "file_name": "shrek-radosne.mp3",
        "text": "Marian! Tu jest jakby luksusowo! Widzisz to słońce? Czuję, że dzisiaj wydarzy się coś absolutnie wspaniałego. Aż chce się żyć, tańczyć i śpiewać!"
    },
    {
        "desc": "Informacyjne / Neutralne",
        "file_name": "shrek-informacja.mp3",
        "text": "Pociąg Intercity relacji Warszawa Centralna – Gdynia Główna wjedzie na tor drugi przy peronie trzecim. Prosimy o zachowanie ostrożności i odsunięcie się od krawędzi peronu."
    },
    {
        "desc": "Smutne / Melancholijne",
        "file_name": "shrek-smunte.mp3",
        "text": "Umrzeć – tego nie robi się kotu. Bo co ma począć kot w pustym mieszkaniu? Drapać się na ściany? Ocierać o meble? Nic niby tu nie zmienione, a jednak wszystko się pozamieniało."
    },
    {
        "desc": "Tajemnicze / Napięcie",
        "file_name": "shrek-napiecie.mp3",
        "text": "Słyszysz to? To nie był wiatr. Kroki na schodach stają się coraz głośniejsze. Ktoś tu jest... stoi tuż za drzwiami i czeka, aż zgaśnie ostatnia świeca."
    }
]

print("Loading XTTS v2 model...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False).to(device_name)

execution_times = []

print(f"\nStarting Tuned Batch Generation (Continuous Flow)...\n" + "-" * 50)

for i, item in enumerate(scenarios):
    print(f"Generating [{i + 1}/{len(scenarios)}]: {item['desc']}")

    ref_path = os.path.join(AUDIO_SAMPLES_DIR, item['file_name'])

    if not os.path.exists(ref_path):
        print(f" ERROR: Could not find '{ref_path}'. Skipping.")
        execution_times.append(0)
        continue

    safe_desc = item['desc'].split("/")[0].strip().replace(" ", "_")
    filename = f"{OUTPUT_DIR}/{i + 1}_{safe_desc}.wav"

    start_time = time.time()

    try:
        tts.tts_to_file(
            text=item["text"],
            speaker_wav=ref_path,
            language="pl",
            file_path=filename,
            split_sentences=False,
            temperature=0.7,
            repetition_penalty=2.0
        )
        success = True
    except Exception as e:
        print(f"Error: {e}")
        success = False

    end_time = time.time()
    duration = end_time - start_time
    execution_times.append(duration if success else 0)

    if use_gpu:
        torch.cuda.empty_cache()

    if success:
        print(f" -> Finished in {duration:.2f}s. Saved: {filename}")
    print("-" * 20)

# --- FINAL SUMMARY ---
print("\n" + "=" * 80)
print("PERFORMANCE SUMMARY (XTTS Tuned)")
print("=" * 80)
print(f"{'#':<4} {'Time (s)':<10} {'Emotion/Type':<25} {'Ref File'}")
print("-" * 80)

for i, (duration, item) in enumerate(zip(execution_times, scenarios)):
    print(f"{i + 1:<4} {duration:<10.2f} {item['desc']:<25} {item['file_name']}")

print("-" * 80)
total_time = sum(execution_times)
print(f"Całkowity czas:     {total_time:.2f} seconds")
if len(scenarios) > 0:
    print(f"Średni czas na klip: {total_time / len(scenarios):.2f} seconds")
print("=" * 80)
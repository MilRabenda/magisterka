import torch
import numpy as np
import soundfile as sf
from transformers import VitsModel, AutoTokenizer
import time
import os

# --- GPU Setup ---
device="cpu"
print(f"Using device: {device}")
if device == "cpu":
    print("WARNING: No GPU detected. Inference might be slower.")

print("Loading Meta MMS (facebook/mms-tts-pol)...")
model_name = "facebook/mms-tts-pol"

model = VitsModel.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# --- Scenarios ---
scenarios = [
    {
        "desc": "Poważne / Podniosłe",
        "speaker": "Meta MMS (Single Speaker)",
        "text": "Litwo! Ojczyzno moja! ty jesteś jak zdrowie. Ile cię trzeba cenić, ten tylko się dowie, Kto cię stracił. Dziś piękność twą w całej ozdobie, Widzę i opisuję, bo tęsknię po tobie."
    },
    {
        "desc": "Radosne / Entuzjastyczne",
        "speaker": "Meta MMS (Single Speaker)",
        "text": "Marian! Tu jest jakby luksusowo! Widzisz to słońce? Czuję, że dzisiaj wydarzy się coś absolutnie wspaniałego. Aż chce się żyć, tańczyć i śpiewać!"
    },
    {
        "desc": "Informacyjne / Neutralne",
        "speaker": "Meta MMS (Single Speaker)",
        "text": "Pociąg Intercity relacji Warszawa Centralna – Gdynia Główna wjedzie na tor drugi przy peronie trzecim. Prosimy o zachowanie ostrożności i odsunięcie się od krawędzi peronu."
    },
    {
        "desc": "Smutne / Melancholijne",
        "speaker": "Meta MMS (Single Speaker)",
        "text": "Umrzeć – tego nie robi się kotu. Bo co ma począć kot w pustym mieszkaniu? Drapać się na ściany? Ocierać o meble? Nic niby tu nie zmienione, a jednak wszystko się pozamieniało."
    },
    {
        "desc": "Tajemnicze / Napięcie",
        "speaker": "Meta MMS (Single Speaker)",
        "text": "Słyszysz to? To nie był wiatr. Kroki na schodach stają się coraz głośniejsze. Ktoś tu jest... stoi tuż za drzwiami i czeka, aż zgaśnie ostatnia świeca."
    }
]

execution_times = []
output_dir = "mms_outputs_cpu"
os.makedirs(output_dir, exist_ok=True)

print(f"\nStarting Batch Generation for {len(scenarios)} scenarios...\n" + "-" * 50)

for i, item in enumerate(scenarios):
    print(f"Generating [{i + 1}/{len(scenarios)}]: {item['desc']}")

    start_time = time.time()

    inputs = tokenizer(item["text"], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model(**inputs).waveform

    end_time = time.time()
    duration = end_time - start_time
    execution_times.append(duration)

    audio_np = output.cpu().numpy().squeeze()

    safe_desc = item['desc'].split("/")[0].strip().replace(" ", "_")
    filename = f"{output_dir}/{i + 1}_{safe_desc}.wav"

    sf.write(filename, audio_np, model.config.sampling_rate)

    print(f" -> Finished in {duration:.2f}s. Saved: {filename}")
    print("-" * 20)

# --- Final Summary ---
print("\n" + "=" * 60)
print("PERFORMANCE SUMMARY (Meta MMS - Polish)")
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
from transformers import VitsModel, AutoTokenizer
import soundfile as sf
import numpy as np
import torch
import os


def tts(text, region):
    # === DIFFERENT MODELS ===
    default="facebook/mms-tts-quz"  
    hg_models = {  # -> different quechua variations
        "san-martin": "facebook/mms-tts-qvs",
        "cuzco": "facebook/mms-tts-quz",
        "huallaga": "facebook/mms-tts-qub",
        "lambayeque": "facebook/mms-tts-quf",
        "south-bolivia": "facebook/mms-tts-quh",   
        "north-bolivia": "facebook/mms-tts-qul",
        "tena-lowland": "facebook/mms-tts-quw",
        "ayacucho": "facebook/mms-tts-quy",
        "cajamarca": "facebook/mms-tts-qvc",
        "eastern-apurimac": "facebook/mms-tts-qve",
        "huamelies": "facebook/mms-tts-qvh",
        "margos-lauricocha": "facebook/mms-tts-qvm",
        "north-junin": "facebook/mms-tts-qvn",
        "huaylas": "facebook/mms-tts-qwh",
        "panao": "facebook/mms-tts-qxh",
        "northern-conchucos": "facebook/mms-tts-qxn",
        "southern-conchucos": "facebook/mms-tts-qxo",
    }

    # === LOAD THE HUGGING FACE MODELS ===
    selected_quechua = hg_models.get(region, default)  
    model = VitsModel.from_pretrained(selected_quechua)  # -> loads the model itself
    tokenizer = AutoTokenizer.from_pretrained(selected_quechua)   # ... token

    # === GENERATE THE AUDIO FILE ===
    inputs = tokenizer(text, return_tensors="pt") 

    with torch.no_grad():  # -> ensures no gradients (as only inference)
        output = model(**inputs).waveform.cpu().numpy()  # -> generates the audio w model and inputs
    output = output / np.max(np.abs(output))  # -> normalizes the waveform

    # === SAVE THE AUDIO FILE ===
    os.makedirs("data", exist_ok=True)
    filenames = [f for f in os.listdir("data") if f.endswith('.wav')]
    if filenames:
        latest_num = max([int(f.split('.')[0].split('_')[1]) for f in filenames])
        filename = f"audio_{latest_num + 1}.wav"
    else:
        filename = "audio_default.wav"

    filepath = os.path.join("data", filename)
    rate = int(model.config.sampling_rate)
    sf.write(filepath, output.T, rate, subtype='PCM_16')
    print(f"\nAudio file saved at: {filepath}")


# === MAIN ===
text = "Libre kanchis, librepuni kasun"
tts(text, "cuzco")

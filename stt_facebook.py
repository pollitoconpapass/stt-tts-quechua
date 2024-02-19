from transformers import Wav2Vec2ForCTC, AutoProcessor
import torchaudio
import torch


def stt(wave_file_path):
    model_id = "facebook/mms-1b-all"
    processor = AutoProcessor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id)

    audio_data, sampling_rate = torchaudio.load(wave_file_path) 
    inputs = processor(audio_data.numpy(), sampling_rate=sampling_rate, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs).logits

    ids = torch.argmax(outputs, dim=-1)[0]
    transcription = processor.decode(ids)
    print(f"\nText from the audio: {transcription}")


# === MAIN ===
stt("data/audio_4.wav")

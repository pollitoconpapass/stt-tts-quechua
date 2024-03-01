from transformers import Wav2Vec2ForCTC, AutoProcessor
import torchaudio
import torch


def stt(wave_file_path):  
    model_id = "facebook/mms-1b-all"
    processor = AutoProcessor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id)

    audio_data, original_sampling_rate = torchaudio.load((wave_file_path))
    resampled_audio_data = torchaudio.transforms.Resample(original_sampling_rate, 16000)(audio_data)
    inputs = processor(resampled_audio_data.numpy(), sampling_rate=16000, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs).logits

    ids = torch.argmax(outputs, dim=-1)[0]
    transcription = processor.decode(ids)
    print(f"\nText from the audio: {transcription}")


# === MAIN ===
stt("data/audio_4.wav")  

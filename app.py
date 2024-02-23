import io
import torch
import torchaudio
import numpy as np
import soundfile as sf
from jsonschema import validate
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from transformers import VitsModel, AutoTokenizer
from transformers import Wav2Vec2ForCTC, AutoProcessor


app = FastAPI()

# === QUECHUA REGIONS SUPPORTED ===
hg_models = {
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
audio_count = 0

# === TEXT-TO-SPEECH CAPABILITY ===
@app.post("/tts-quechua")
async def tts(data: dict):
    global audio_count
    try:
        schema = {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "region": {"type": "string"},
                "audio-name": {"type": "string"},
            },
            "required": ["text", "region"]
        }

        validate(instance=data, schema=schema)

        text = data["text"]
        region = data["region"]
        audio_name = data.get("audio-name")

        default="facebook/mms-tts-quz"
        selected_quechua = hg_models.get(region, default)  
        model = VitsModel.from_pretrained(selected_quechua)  # -> loads the model itself
        tokenizer = AutoTokenizer.from_pretrained(selected_quechua)   # ... token
    
        inputs = tokenizer(text, return_tensors="pt") 

        with torch.no_grad():  # -> ensures no gradients (as only inference)
            output = model(**inputs).waveform.cpu().numpy()  # -> generates the audio w model and inputs
        output = output / np.max(np.abs(output))  # -> normalizes the waveform

        rate = int(model.config.sampling_rate)

        if audio_name is None or audio_name == "":
            audio_count += 1
            audio_name = f"audio_{audio_count}"

        file_path = f"audios/{audio_name}.wav"
        sf.write(file_path, output.T, rate, subtype='PCM_16', format='WAV')

        with open(file_path, "rb") as audio_file:
            audio_bytes = audio_file.read()

        return StreamingResponse(io.BytesIO(audio_bytes), media_type="audio/wav")
    
    except Exception as e:
        return {"error": str(e)}


# === SPEECH-TO-TEXT CAPABILITY ===
@app.post("/stt-quechua")
async def stt(wav_file: UploadFile = File(...)):
    try:
        model_id = "facebook/mms-1b-all"
        processor = AutoProcessor.from_pretrained(model_id)
        model = Wav2Vec2ForCTC.from_pretrained(model_id)

        audio_data, original_sampling_rate = torchaudio.load(io.BytesIO(await wav_file.read()))
        resampled_audio_data = torchaudio.transforms.Resample(original_sampling_rate, 16000)(audio_data)
        inputs = processor(resampled_audio_data.numpy(), sampling_rate=16000, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs).logits

        ids = torch.argmax(outputs, dim=-1)[0]
        transcription = processor.decode(ids)

        return {"transcription": transcription}
    
    except Exception as e:
        return {"error": str(e)}
    

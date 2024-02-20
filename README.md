# stt-tts-quechua

### Usage: 
1. Run ```uvicorn app:app --reload```
2. Copy the link you got and paste it into a Postman request. eg: http://127.0.0.1:8000
3. Add `/stt-quechua` or `/tts-quechua` for the scenario you prefer.


 ### Functionalities
 Speech-to-Text:
 - You need a wav file in Quechua language already stored in your local
 - Upload the audio file like this:
      1. Body -> form-data
      2. Choose Text, type: wav_file
      3. In the same cell, change to File
      4. In the value part upload your wav file
         
   ![image](https://github.com/pollitoconpapass/stt-tts-quechua/assets/90667035/ddbc9e71-6813-41e1-b877-24d675e13e1a)


Text-to-Speech:
- Here it will be a bit more simple
- Go to Body -> raw -> JSON
- Requested values: `text` and `region` (both strings)
- Check the supported regions on the master's README

  ![image](https://github.com/pollitoconpapass/stt-tts-quechua/assets/90667035/6f2e174c-0d3b-4503-8755-50725a0140fa)

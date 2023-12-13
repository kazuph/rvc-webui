import io
import json
import os
import traceback
from typing import *
import numpy as np

import soundfile as sf
from flask import Flask, make_response, request, send_file
from scipy.io.wavfile import write

from modules.server.model import VoiceServerModel

model: Optional[VoiceServerModel] = None
app = Flask(__name__)

@app.route('/ping')
def ping():
    return make_response("server is alive", 200)

@app.route('/upload_model', methods=['POST'])
def upload_model():
    """
    input:
        json:
            rvc_model_file: str
                specify rvc model's absolute path (.pt, .pth)
            faiss_index_file: Optional[str]
                specify faiss index'S absolute path (.index)
    """
    global model
    if request.method == "POST":
        rvc_model_file = request.json["rvc_model_file"]
        faiss_index_file =request.json["faiss_index_file"] if "faiss_index_file" in request.json else ""
        try:
            model = VoiceServerModel(rvc_model_file, faiss_index_file)
            return make_response("model is load", 200)
        except:
            traceback.print_exc()
            return make_response("model load error", 400)
    else:
        return make_response("use post method", 400)

@app.route('/convert_sound', methods=['POST'])
def convert_sound():
    """
    input:
        form:
            model_name: str (optional)
                specify the model name
            speaker_id: int (optional)
                default: 0
            transpose: int (optional)
                default: 0
            pitch_extraction_algo: str (optional)
                default: dio
                value: ["dio", "harvest", "mangio-crepe", "crepe"]
            retrieval_feature_ratio: float (optional)
                default: 0
                value: 0. ~ 1.
        input_wav: wav file

    output:
        wavfile

    curl:
        ```
        $ curl -X POST -F "input_wav=@./input.mp3" http://localhost:5001/convert_sound \
            -F "model_name=<YOUR_MODEL_NAME>.pth" \
            >| ./outputs/output.wav
        ```

    """
    print("start")
    if request.method == "POST":
        input_buffer = io.BytesIO(request.files["input_wav"].stream.read())
        audio, sr = sf.read(input_buffer)

        # stereo -> mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        model_name = request.form.get("model_name", "")
        sid = int(request.form.get("speaker_id", 0))
        transpose = int(request.form.get("transpose", 0))
        pitch_extraction_algo = request.form.get("pitch_extraction_algo", "dio")
        if pitch_extraction_algo not in ["dio", "harvest", "mangio-crepe", "crepe"]:
            return make_response("bad pitch extraction algo", 400)
        retrieval_feature_ratio = float(request.form.get("retrieval_feature_ratio", 0.))

        model = VoiceServerModel(f"/app/models/checkpoints/{model_name}", "")
        out_audio = model(audio, sr, sid, transpose, pitch_extraction_algo, retrieval_feature_ratio)
        output_buffer = io.BytesIO()
        write(output_buffer, rate=model.tgt_sr, data=out_audio)
        output_buffer.seek(0)
        response = make_response(send_file(output_buffer, mimetype="audio/wav"), 200)
        return response
    else:
        return make_response("use post method", 400)

if __name__ == "__main__":
    # app.run()
    app.run(host='0.0.0.0', port=5001)

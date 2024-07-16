import base64
import io
import json
import logging
import numpy as np
import os
import requests
import re

from PIL import Image
from flask import Flask, jsonify, request

application = Flask(__name__)

INFERENCE_ENDPOINT = os.environ.get("INFERENCE_ENDPOINT") or "http://modelmesh-serving.mnist.svc.cluster.local:8008/v2/models/mnist/infer"
OAUTH_TOKEN = os.environ.get("OAUTH_TOKEN")

def preprocess(img_b64):
    if img_b64.startswith("data:image/"):
        img_b64 = re.search(r'base64,(.*)', img_b64).group(1)
    image_bytes = io.BytesIO(base64.b64decode(img_b64))
    im = Image.open(image_bytes)
    im = im.resize((28, 28))
    arr = np.array(im)[:,:,0:1]
    arr = (255 - arr) / 255
    res = arr.reshape(1,-1).tolist()
    return res

def predict(data):
    f = open("payload_template.json")
    payload = json.load(f)
    payload['inputs'][0]['data'] = data
    application.logger.debug(f"Payload: {payload}")
    payload = json.dumps(payload)
    r = requests.post(INFERENCE_ENDPOINT, data = payload, headers = headers ,verify = False)
    application.logger.info(f"Model server responded with code: {r.status_code}")
    inference = r.json()
    return inference

def postprocess(payload):
    prediction = payload['outputs'][0]['data']
    return prediction

@application.route('/', methods=['GET'])
def status():
    return jsonify({'status': 'ok'})

@application.route('/', methods=['POST'])
def inference():
    img_b64 = request.json["png"]
    application.logger.debug(f"PNG in base64: {img_b64}")
    data = preprocess(img_b64)
    payload = predict(data[0])
    print(payload)
    prediction = postprocess(payload)
    return jsonify({"data": prediction})

if __name__ == '__main__':
    headers = {}
    if OAUTH_TOKEN != None:
        headers['Authorization'] = f'Bearer {OAUTH_TOKEN}'
    application.logger.setLevel(logging.INFO)
    application.run(host='0.0.0.0',port=8080)
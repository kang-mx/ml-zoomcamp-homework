import json
import numpy as np
import onnxruntime as ort
from io import BytesIO
from urllib import request
from PIL import Image

def download_image(url: str) -> Image.Image:
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def prepare_image(img: Image.Image, target_size: tuple) -> Image.Image:
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def preprocess(img: Image.Image) -> np.ndarray:
    x = np.array(img).astype("float32") / 255.0
    x = (x - 0.5) * 2
    x = np.transpose(x, (2, 0, 1))  # HWC -> CHW
    x = np.expand_dims(x, 0)        # add batch dimension
    return x

session = ort.InferenceSession("hair_classifier_empty.onnx")
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Read model shape dynamically
model_shape = session.get_inputs()[0].shape
# Convert batch placeholder to 1
target_height = int(model_shape[2])
target_width = int(model_shape[3])
print(f"DEBUG: Model expects input shape: {model_shape}, using target size: ({target_height},{target_width})")

def lambda_handler(event, context=None):
    url = event.get("url")
    if not url:
        return {"error": "No 'url' key found in event"}

    # Prepare image
    img = download_image(url)
    img = prepare_image(img, target_size=(target_height, target_width))
    x = preprocess(img)

    # Debug: print input shape
    print("DEBUG: x.shape =", x.shape)

    # Run model
    pred = session.run([output_name], {input_name: x})[0]
    score = float(pred[0][0])
    return {"prediction": score}
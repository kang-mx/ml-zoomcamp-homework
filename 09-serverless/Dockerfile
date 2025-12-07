FROM agrigorev/model-2025-hairstyle:v1

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY lambda_handler.py .
COPY hair_classifier_empty.onnx .
COPY hair_classifier_v1.onnx.data . 

CMD ["lambda_handler.lambda_handler"]

import json
import numpy as np
import cv2
import tensorflow as tf




def load_model(model_path):
    return tf.keras.models.load_model(model_path)




def load_labels(labels_path):
    with open(labels_path, 'r') as f:
        return json.load(f)




def preprocess_image_bytes(image_bytes, image_size=150):
# decode bytes -> OpenCV image (BGR)
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError('Could not decode image bytes')
# convert BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size, image_size))
    img = img.astype('float32') / 255.0
    return np.expand_dims(img, axis=0)




def predict_from_bytes(image_bytes, model, labels, image_size=150, top_k=3):
    x = preprocess_image_bytes(image_bytes, image_size)
    preds = model.predict(x)[0]
    idxs = preds.argsort()[::-1][:top_k]
    results = [{'label': labels[i], 'probability': float(preds[i])} for i in idxs]
    return results




# quick CLI test
if __name__ == '__main__':
    import sys
    if len(sys.argv) < 4:
        print('Usage: python src/predict.py model_path labels.json image.jpg')
        sys.exit(1)
    model = load_model(sys.argv[1])
    labels = load_labels(sys.argv[2])
    with open(sys.argv[3], 'rb') as f:
        b = f.read()
    print(predict_from_bytes(model, b, labels))
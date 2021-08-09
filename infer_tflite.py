import time
import numpy as np
import tensorflow as tf
from PIL import Image

def load_labels(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]

interpreter = tf.lite.Interpreter(model_path='./models/tf_export/pt_mobilenet/model_float32.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

is_floating_model = input_details[0]['dtype'] == np.float32

labels = load_labels('./labels.txt')
# img = Image.open('./images/grace_hopper.bmp').resize((224, 224))
# img = Image.open('./images/n01443537_goldfish.JPEG').resize((224, 224))
img = Image.open('./images/n02099601_golden_retriever.JPEG').resize((224, 224))

input_data = np.expand_dims(img, axis=0)
if is_floating_model:
    # TensorFlow models normalization
    # input_data = (np.float32(input_data) - 127.5) / 127.5

    # Torchvision models normalization
    input_data = np.float32(input_data) / 255.0
    input_data = (input_data - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)

interpreter.set_tensor(input_details[0]['index'], input_data)

start_time = time.time()
interpreter.invoke()
stop_time = time.time()

output_data = interpreter.get_tensor(output_details[0]['index'])
results = np.squeeze(output_data)

top_k = results.argsort()[-5:][::-1]
for i in top_k:
    if is_floating_model:
        print("{:08.6f}: {}".format(float(results[i]), labels[i]))
    else:
        print("{:08.6f}: {}".format(float(results[i] / 255.0), labels[i]))

print("time: {:.3f}ms".format((stop_time - start_time) * 1000))
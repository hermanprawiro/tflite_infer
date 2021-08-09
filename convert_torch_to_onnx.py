import numpy as np
import onnxruntime as ort
import torch
import torchvision

model = torchvision.models.mobilenet.mobilenet_v2(pretrained=True)
model.eval()
dummy_input = torch.randn(1, 3, 224, 224)

input_names = ['input1']
output_names = ['output1']

torch.onnx.export(model, dummy_input, './models/pt_mobilenet.onnx', verbose=True, input_names=input_names, output_names=output_names)

with torch.no_grad():
    out_torch = model(dummy_input).numpy()

ort_session = ort.InferenceSession('./models/pt_mobilenet.onnx')
out_ort = ort_session.run(None, {input_names[0]: dummy_input.numpy()})[0]

print(np.allclose(out_torch, out_ort, atol=1e-5))
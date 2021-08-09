import onnx
from onnxsim import simplify

"""
python -m onnxsim models/pt_mobilenet.onnx models/pt_mobilenet_opt.onnx
"""

model = onnx.load('./models/pt_mobilenet.onnx')

print('Simplifying...')
model_simp, check = simplify(model)

if check:
    print('Check OK')
    onnx.save(model_simp, './models/pt_mobilenet_opt.onnx')
else:
    print('Check failed')
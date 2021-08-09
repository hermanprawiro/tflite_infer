# PyTorch to TFLite

[Reference](https://qiita.com/PINTO/items/ed06e03eb5c007c2e102)

Steps:
1. PyTorch => ONNX (PyTorch Native ONNX Export Tools)
2. Optimize ONNX (onnx-simplifier)
3. Optimized ONNX => OpenVINO (OpenVINO Model Optimizer)
4. OpenVINO => TF/TFLite (openvino2tensorflow)

## Detailed Steps

### PyTorch to ONNX (NCHW to NCHW)

[PyTorch ONNX Docs](https://pytorch.org/docs/stable/onnx.html)

### Optimize ONNX (NCHW)

[pypi](https://pypi.org/project/onnx-simplifier/) | [GitHub](https://github.com/daquexian/onnx-simplifier)

To install via pip:
```shell
pip install onnx-simplifier
```

To optimize the ONNX model:
```shell
python -m onnxsim models/pt_mobilenet.onnx models/pt_mobilenet_opt.onnx
```

### Optimized ONNX to OpenVINO (NCHW to NCHW)

[OpenVINO Docs](https://docs.openvinotoolkit.org/latest/index.html)

```shell
python "C:\Program Files (x86)\Intel\openvino_2021\deployment_tools\model_optimizer\mo.py" \
    --input_model models\pt_mobilenet_opt.onnx \
    --input_shape [1,3,224,224] \
    --output_dir models\openvino\pt_mobilenet \
    --data_type FP32
```

### OpenVINO to TensorFlow/TensorFlowLite (NCHW to NHWC)

[GitHub](https://github.com/PINTO0309/openvino2tensorflow)

Use Docker environment `pinto0309/openvino2tensorflow`.

```shell
docker run -it --rm -v `pwd`:/home/user/workdir pinto0309/openvino2tensorflow:latest
```

Inside Docker container, run:
```shell
openvino2tensorflow \
    --model_path models/openvino/pt_mobilenet/pt_mobilenet_opt.xml \
    --model_output_path models/tf_export/pt_mobilenet \
    --output_saved_model \
    --output_h5 \
    --output_pb \
    --output_no_quant_float32_tflite \
    --output_weight_quant_tflite \
    --output_float16_quant_tflite
```

## Other Tools

### TFLite Benchmark Tools
[https://www.tensorflow.org/lite/performance/measurement](https://www.tensorflow.org/lite/performance/measurement)
[GitHub](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark)

[Benchmark Tools built with MTK Neuron Delegate](https://github.com/MediaTek-NeuroPilot/tflite-neuron-delegate)

### Approximate FLOPS Calculation (tflite-flops)
[GitHub](https://github.com/lisosia/tflite-flops)

Only considers Conv and DepthwiseConv.

Install:
```shell
pip3 install git+https://github.com/lisosia/tflite-flops
```

Usage:
```
python -m tflite_flops models\tf_export\pt_mobilenet\model_float32.tflite
```

Calculation:
```
Multiply-Accumulate (MAC) = output_h * output_w * output_c * kernel_h * kernel_w * input_c 
                         (= output_h * output_w * weight_size)
Floating-point operations (FLOPs) = 2 * MAC
```
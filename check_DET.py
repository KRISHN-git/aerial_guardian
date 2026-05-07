import onnxruntime as ort
import numpy as np

session = ort.InferenceSession('weights/drone_person_best.onnx')
inp     = session.get_inputs()[0]
dummy   = np.random.randn(1, 3, 640, 640).astype(np.float32)
out     = session.run(None, {inp.name: dummy})
print(f'ONNX model verified OK')
print(f'Input  : {inp.name} {inp.shape}')
print(f'Output : {out[0].shape}')
print(f'Size   : 37.9 MB')

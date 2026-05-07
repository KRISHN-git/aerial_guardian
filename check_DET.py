from ultralytics import YOLO
model = YOLO('weights/drone_person_best.pt')
model.export(format='onnx', opset=11, simplify=False, dynamic=False)
print('ONNX export complete')

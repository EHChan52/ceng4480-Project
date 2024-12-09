from ultralytics import YOLO

model = YOLO("yolov11custom.pt")

model.predict(source = "test2.mp4", show = True, save = True, conf = 0.4)

# change to source = "0"
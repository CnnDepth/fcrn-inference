import cv2

capture = cv2.VideoCapture("nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)I420, framerate=(fraction)120/1 ! \
                            nvvidconv   ! video/x-raw, format=(string)BGRx ! \
                            videoconvert ! video/x-raw, format=(string)BGR ! \
                            appsink")
#capture.set(3, 480)
#capture.set(4, 640)
assert capture.read()[0]

from tqdm import tqdm
frames = []
for i in tqdm(range(25)):
    frame = capture.read()[1]
    frames.append(frame)
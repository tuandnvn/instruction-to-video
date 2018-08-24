from evaluation_utils import crop_and_resize
import cv2

file_path = "../instruction-to-video/target/0/0.mp4"

cap = cv2.VideoCapture(file_path)
_, first_frame = cap.read()
first_frame = crop_and_resize(first_frame)
print(first_frame.shape)

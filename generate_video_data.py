from evaluation_utils import crop_and_resize
import cv2
import pickle
import os
import numpy as np

with open("../instruction-to-video/data/puzzle.dat", "rb") as f:
    puzzle = pickle.load(f)

frames = list()
eval_frames = list()
test_frames = list()

# get image training set and eval set
for num in range(200):
    num_instructions = len(puzzle[num]) - 1
    directory = num // 100
    video_path = os.path.join('..', 'instruction-to-video/target', str(directory), str(num) + '.mp4')
    cap = cv2.VideoCapture(video_path)
    _, first_frame = cap.read()
    first_frame = crop_and_resize(first_frame)
    for _ in range(num_instructions):
        frames.append(first_frame)
    eval_frames.append(first_frame)

# get image testing set
for num in range(200, 300):
    directory = num // 100
    video_path = os.path.join('..', 'instruction-to-video/target', str(directory), str(num) + '.mp4')
    cap = cv2.VideoCapture(video_path)
    _, first_frame = cap.read()
    first_frame = crop_and_resize(first_frame)
    test_frames.append(first_frame)

# convert list to numpy arrays
frames = np.asarray(frames)
eval_frames = np.asarray(eval_frames)
test_frames = np.asarray(test_frames)

# save them as npy files
np.save("../instruction-to-video/data/image_train.npy", frames)
np.save("../instruction-to-video/data/image_eval.npy", eval_frames)
np.save("../instruction-to-video/data/image_test.npy", test_frames)


import cv2
#from keras_video import SlidingFrameGenerator
import os
import glob
from collections import deque
#import keras

classes = [i.split(os.path.sep)[1] for i in glob.glob('videos/*')]
# classes.sort()

# SIZE = (112, 112)
# CHANNELS = 3
# NBFRAME = 5
# BS = 8

# glob_pattern = 'videos/{classname}/*.mp4'
# # for data augmentation
# data_aug = keras.preprocessing.image.ImageDataGenerator(
#     zoom_range=.1,
#     horizontal_flip=True,
#     rotation_range=8,
#     width_shift_range=.2,
#     height_shift_range=.2)

# frameGen = SlidingFrameGenerator(
#     classes=classes,
#     glob_pattern=glob_pattern,
#     nb_frames=NBFRAME,
#     shuffle=True,
#     batch_size=BS,
#     target_shape=SIZE,
#     nb_channel=CHANNELS,
#     transformation=data_aug,
#     use_frame_cache=True)

name = "output.mp4"
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

if (cap.isOpened() == False):
    print("Unable to open baby camera for baby vid")




while(True):
    ret, frame = cap.read()



    if ret == True:
        cv2.imshow('frame', frame)
        out.write(frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break;

    else:
        break

cap.release()
cv2.destroyAllWindows()



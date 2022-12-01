# import cv2
# #from keras_video import SlidingFrameGenerator
# import os
# import glob
# from collections import deque
# #import keras

# classes = [i.split(os.path.sep)[1] for i in glob.glob('videos/*')]
# # classes.sort()

# # SIZE = (112, 112)
# # CHANNELS = 3
# # NBFRAME = 5
# # BS = 8

# # glob_pattern = 'videos/{classname}/*.mp4'
# # # for data augmentation
# # data_aug = keras.preprocessing.image.ImageDataGenerator(
# #     zoom_range=.1,
# #     horizontal_flip=True,
# #     rotation_range=8,
# #     width_shift_range=.2,
# #     height_shift_range=.2)

# # frameGen = SlidingFrameGenerator(
# #     classes=classes,
# #     glob_pattern=glob_pattern,
# #     nb_frames=NBFRAME,
# #     shuffle=True,
# #     batch_size=BS,
# #     target_shape=SIZE,
# #     nb_channel=CHANNELS,
# #     transformation=data_aug,
# #     use_frame_cache=True)

# name = "output.mp4"
# cap = cv2.VideoCapture(0)

# if (cap.isOpened() == False):
#     print("Unable to open baby camera for baby vid")


# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))

# while(True):
#     ret, frame = cap.read()



#     if ret == True:
#         cv2.imshow('frame', frame)


#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break;

#     else:
#         break

# cap.release()
# cv2.destroyAllWindows()


import os
import glob
import cv2
import numpy as np
from collections import deque


from tensorflow.keras.models import load_model


model = load_model("./saved_models/mobilenet_1.h5")
classes = [i.split(os.path.sep)[1] for i in glob.glob('videos/*')]


mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
queue = deque(maxlen=128)

# start running cell from here till the end after loading the model
cap = cv2.VideoCapture(0)

writer = None
(W, H) = (None, None)
buffer = []
old_label= None
while(True):
    
    while (len(buffer) < 5):
        grabbed, frame = cap.read()
    
        if not grabbed:
            break
        if(old_label is not None):
            text = "state: {}".format(old_label)
            cv2.putText(frame, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,1.25, (0, 255, 0), 5)
        cv2.imshow("Main detection", frame)
        key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (112, 112)).astype("float32")
        buffer.append(frame)
    
    if key == ord("q"):
        break
        
    print("Predicting sequence")
    try:
        preds = model.predict(np.expand_dims(np.asarray(buffer),axis=0))
        buffer.clear()
	# perform prediction averaging over the current history of
        queue.append(preds)
    except:
        print("crashed")
        break
    
    results = np.array(queue).mean(axis=0)
    print (results)
    i = np.argmax(results)
    print (i)
    label = classes[i]
    old_label=label
    
    buffer.clear()

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
import os
import glob
import keras_video.utils
from tensorflow import keras
from keras_video import VideoFrameGenerator


# use sub directories names as classes
classes = [i.split(os.path.sep)[1] for i in glob.glob('videos/*')]
classes.sort()
# some global params
SIZE = (112, 112)
CHANNELS = 3
NBFRAME = 5
BS = 8
# pattern to get videos and classes
glob_pattern = 'videos/{classname}/*.mp4'
# for data augmentation
data_aug = keras.preprocessing.image.ImageDataGenerator(
    zoom_range=.1,
    horizontal_flip=True,
    rotation_range=8,
    width_shift_range=.2,
    height_shift_range=.2)
# Create video frame generator
test = VideoFrameGenerator(
    classes=classes,
    glob_pattern=glob_pattern,
    nb_frames=NBFRAME,
    shuffle=True,
    batch_size=BS,
    target_shape=SIZE,
    nb_channel=CHANNELS,
    transformation=data_aug,
    use_frame_cache=True)

# Showing a sample shows some of the generated frame it is not required
# keras_video.utils.show_sample(train)

new_model = keras.models.load_model('saved_models/covnet_1.h5')

# Evaluate allows us to run the full saved model and get simple results of test loss, test acc
print("Evaluate on test data")
results = new_model.evaluate(test)
print("test loss, test acc:", results)

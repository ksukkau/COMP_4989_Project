import os
import glob
import keras_video.utils
from tensorflow import keras
from keras_video import VideoFrameGenerator, SlidingFrameGenerator

# TODO process test data without adding class tags
# use sub-directories names as classes
# there won't be any classes in actual prediction data
classes = [i.split(os.path.sep)[1] for i in glob.glob('videos/*')]
classes.sort()
# some global params
SIZE = (112, 112)
CHANNELS = 3
NBFRAME = 5
BS = 2
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
    # classes=classes,
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

# printing the files sent in for comparison to output
print(test.files)
model = keras.models.load_model('saved_models/mobilenet_sliding.h5')


print("Evaluate on test data")
results = model.evaluate(test)
print("test loss, test acc:", results)

print("Generate predictions")
predictions = model.predict(test)
# the prediction gives a row for each sample and the likely-hood of it being each of the 24 classes.
print("predictions shape:", predictions.shape)

# TODO display images to verify correctness?
for prediction in predictions:
    predicted_class = ''
    max_val = 0
    for i in range(len(prediction)):
        if prediction[i] > max_val:
            max_val = prediction[i]
            predicted_class = classes[i]

    print(predicted_class)

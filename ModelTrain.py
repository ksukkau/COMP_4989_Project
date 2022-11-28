"""
Sources:
https://medium.com/smileinnovation/training-neural-network-with-image-sequence-an-example-with-video-as-input-c3407f7a0b0f
https://medium.com/iitg-ai/how-to-use-callbacks-in-keras-to-visualize-monitor-and-improve-your-deep-learning-model-c9ca37901b28
"""
import os
import glob
import keras
import keras_video.utils
from tensorflow import keras
# to use VideoFrameGenerator you must edit imports in the generator.py file, or it will not work
# env/Lib/site-packages/keras_video/generator.py
# from keras.preprocessing.image import ImageDataGenerator
# from keras.utils import img_to_array
from keras_video import VideoFrameGenerator, SlidingFrameGenerator
from keras.layers import Conv2D, BatchNormalization, \
    MaxPool2D, GlobalMaxPool2D
from keras.layers import TimeDistributed, GRU, Dense, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard

# Stops model if there is no change to val_loss after 3 epochs, keeps the west weights
# Seems to stop too early
earlystop = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=3,
                          verbose=1,
                          restore_best_weights=True)

# creates tensorboard visualizations to view results
# to view run in terminal: tensorboard --logdir=sliding_logs
tbCallBack = TensorBoard(log_dir="sliding_logs",
                         histogram_freq=0,
                         write_graph=True,
                         write_images=False)

# reduces learning rate when a metric stops improving
reduceLRO = ReduceLROnPlateau(verbose=1)

# ModelCheckpoint creates models inside given directory
modelCheckpoint = ModelCheckpoint(
    'chkp/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
    verbose=1),

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
# Create video frame generator/sliding frame generator
train = SlidingFrameGenerator(
    classes=classes,
    glob_pattern=glob_pattern,
    nb_frames=NBFRAME,
    split=.33,
    shuffle=True,
    # batch_size=BS,
    target_shape=SIZE,
    nb_channel=CHANNELS,
    transformation=data_aug,
    use_frame_cache=True)

valid = train.get_validation_generator()

keras_video.utils.show_sample(train)


# This model does feature detection and uses GlobalMaxPool2D which reduces
# the number of outputs getting only maximum values from the last convolution
# convnet == CNN
def build_convnet(shape=(112, 112, 3)):
    momentum = .9
    model = keras.Sequential()
    model.add(Conv2D(64, (3, 3), input_shape=shape,
                     padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=momentum))

    model.add(MaxPool2D())

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=momentum))

    model.add(MaxPool2D())

    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=momentum))

    model.add(MaxPool2D())

    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=momentum))

    # flatten...
    model.add(GlobalMaxPool2D())
    return model


# This is the time distributed model where we add the dimension of 5 frames,
# the time distributed layer gets 5 images of size 112x112 with 3 channels (RGB)
# action_model calls build_convnet and adds that to the time distributed model
def action_model(shape=(5, 112, 112, 3), nbout=3):
    # Create our convnet with (112, 112, 3) input shape
    convnet = build_convnet(shape[1:])

    # then create our final model
    model = keras.Sequential()
    # add the convnet with (5, 112, 112, 3) shape
    model.add(TimeDistributed(convnet, input_shape=shape))
    # here, you can also use GRU or LSTM
    model.add(GRU(64))
    # and finally, we make a decision network
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(nbout, activation='softmax'))
    return model


INSHAPE = (NBFRAME,) + SIZE + (CHANNELS,)  # (5, 112, 112, 3)
model = action_model(INSHAPE, len(classes))
optimizer = keras.optimizers.Adam(0.001)

model.compile(
    optimizer,
    'categorical_crossentropy',
    metrics=['acc']
)

EPOCHS = 50

callbacks = [
    modelCheckpoint,
    reduceLRO,
    # earlystop,
    tbCallBack
]

# assigning to history saves a record of the values
history = model.fit(
    train,
    validation_data=valid,
    verbose=1,
    epochs=EPOCHS,
    callbacks=callbacks
)

# prints metric values
# print(history.history)

model.save('saved_models/convnet_sliding.h5')

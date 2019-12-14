import cv2
import numpy as np
import glob
import random
from keras.models import Model, Sequential
from keras.layers import Input, Activation, Add, Reshape, Permute
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, UpSampling2D
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from prepareData import Data_Generator, getImageArray
K.set_image_dim_ordering("th")  # channel first

def displayImage(real,predicted):
    #cv2.imshow("Real",real)
    cv2.imshow("Predicted",predicted)
    cv2.waitKey(0)  
    cv2.destroyAllWindows()
    
def visualize(filename):
    test_image = getImageArray(filename)
    test_image = np.expand_dims(test_image, axis=0)
    predicted_image = model.predict(test_image)[0]
    predicted_image = predicted_image.reshape((img_height, img_width, 3)).argmax(axis=2)
    
    seg_img = np.zeros((img_height, img_width, 3))
    colors = [(0, 0, 0),(0, 255, 0),(0, 0, 255)]
    for c in range(3):
        seg_img[:,:,0] += ((predicted_image[:,: ] == c)*(colors[c][0])).astype('uint8')
        seg_img[:,:,1] += ((predicted_image[:,: ] == c)*(colors[c][1])).astype('uint8')
        seg_img[:,:,2] += ((predicted_image[:,: ] == c)*(colors[c][2])).astype('uint8')
    
    seg_img = cv2.resize(seg_img, (img_width, img_height))	 
    displayImage(test_image,seg_img)
    path = "results\\"+filename.split('/')[-1]
    cv2.imwrite(path,seg_img)

def residual_block(x):
    m = Conv2D(8,(1,1),activation='relu',padding="same")(x)
    m = Conv2D(8,(5,1),activation='relu',padding="same")(m)
    m = Conv2D(8,(1,5),activation='relu',padding="same")(m)
    m = Conv2D(16,(1,1),activation='relu',padding="same")(m)
    return Add()([m, x])

img_height = 384
img_width = 512

batch_size = 5
# Load the training data
filenames = glob.glob("rgb_resized/*.png")
filenames.sort()  # make sure that the filenames have a fixed order before shuffling
random.seed(230)
random.shuffle(filenames)

split_1 = int(0.8 * len(filenames))
split_2 = int(0.9 * len(filenames))
train_filenames = filenames[:split_1]
validation_filenames = filenames[split_1:split_2]
test_filenames = filenames[split_2:]    
gt_training = ["annotation\\"+filename.split('\\')[-1] for filename in train_filenames]
gt_validation = ["annotation\\"+filename.split('\\')[-1] for filename in validation_filenames]
gt_testing = ["groundtruth_resized\\"+filename.split('\\')[-1] for filename in test_filenames]

training_batch_generator = Data_Generator(train_filenames, gt_training, batch_size)
validation_batch_generator = Data_Generator(validation_filenames, gt_validation, batch_size)



# Model
input_shape = (14,img_height,img_width)
i = Input(shape=input_shape)
# Encoder
conv_1 = Conv2D(16,(5,5),padding="same")(i)
conv_1 = BatchNormalization()(conv_1)
conv_1 = Activation('relu')(conv_1)
rb_1 = residual_block(conv_1)
rb_2 = residual_block(rb_1)
rb_3 = residual_block(rb_2)
pool_1 = MaxPooling2D(pool_size=(2, 2))(rb_3)

rb_4 = residual_block(pool_1)
rb_5 = residual_block(rb_4)
rb_6 = residual_block(rb_5)
pool_2 = MaxPooling2D(pool_size=(2, 2))(rb_6)

rb_7 = residual_block(pool_2)
rb_8 = residual_block(rb_7)
rb_9 = residual_block(rb_8)
pool_3 = MaxPooling2D(pool_size=(2, 2))(rb_9)

rb_10 = residual_block(pool_3)
rb_11 = residual_block(rb_10)
rb_12 = residual_block(rb_11)
pool_4 = MaxPooling2D(pool_size=(2, 2))(rb_12)

# Decoder
upsamp_1 = UpSampling2D(size=(2,2))(pool_4)
rb_13 = residual_block(upsamp_1)
rb_14 = residual_block(rb_13)
rb_15 = residual_block(rb_14)

upsamp_2 = UpSampling2D(size=(2,2))(rb_15)
rb_16 = residual_block(upsamp_2)
rb_17 = residual_block(rb_16)
rb_18 = residual_block(rb_17)

upsamp_3 = UpSampling2D(size=(2,2))(rb_18)
rb_19 = residual_block(upsamp_3)
rb_20 = residual_block(rb_19)
rb_21 = residual_block(rb_20)

upsamp_4 = UpSampling2D(size=(2,2))(rb_21)
rb_22 = residual_block(upsamp_4)
rb_23 = residual_block(rb_22)
rb_24 = residual_block(rb_23)

conv_2 = Conv2D(3,(1,1),padding="same")(rb_24)
conv_2 = Reshape((3,img_height*img_width), input_shape=(3,img_width,img_height))(conv_2)
conv_2 = Permute((2, 1))(conv_2)
j = Activation('softmax')(conv_2)

model = Model(inputs=i, outputs=j) 
model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
checkpoint = ModelCheckpoint('model.h5',verbose = 1,monitor = 'acc',save_best_only = True,mode = "auto")
model.fit_generator(generator=training_batch_generator,
                    steps_per_epoch=(len(train_filenames) // batch_size),
                    epochs=1,
                    verbose=1,
                    validation_data=validation_batch_generator,
                    validation_steps=(len(validation_filenames) // batch_size),callbacks = [checkpoint])
                    
#model.save_weights('model.h5')
#model.load_weights('model.h5')

# Evaluating
x_test, y_test = next(load_data("test/" , "test_masks/" ,  21, 3 , img_height , img_width , img_height , img_width))
score = model.evaluate(x_test[:10],y_test[:10])

visualize(getImageArray(test_filenames[15]))

# Testing Data
filenames = glob.glob("test_annotations/*.png")
filenames.sort()
test_annotations = [cv2.imread(img) for img in filenames]
test_annotations = [cv2.resize(img,(img_width,img_height),interpolation = cv2.INTER_NEAREST) for img in test_annotations]


# Segnet Model
'''
kernel = 3
filter_size = 64
pad = 1
pool_size = 2

model = Sequential()
# encoder
model.add(Conv2D(filter_size, (kernel, kernel), padding = 'same', input_shape=(14, img_height , img_width)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

model.add(Conv2D(128, (kernel, kernel), padding = 'same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

model.add(Conv2D(256, (kernel, kernel), padding = 'same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

model.add(Conv2D(512, (kernel, kernel), padding = 'same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

# decoder
model.add(Conv2D(512, (kernel, kernel), padding = 'same'))
model.add( BatchNormalization())

model.add(UpSampling2D(size=(pool_size,pool_size)))
model.add(Conv2D(256, (kernel, kernel),  padding = 'same'))
model.add(BatchNormalization())

model.add(UpSampling2D(size=(pool_size,pool_size)))
model.add(Conv2D(128, (kernel, kernel),  padding = 'same'))
model.add(BatchNormalization())

model.add(UpSampling2D(size=(pool_size,pool_size)))
model.add(Conv2D(filter_size, (kernel, kernel), padding = 'same'))
model.add(BatchNormalization())
 
model.add(Conv2D(3, (1, 1), padding = 'same'))
model.add(Reshape((3, model.output_shape[-2]*model.output_shape[-1]), input_shape=(3, model.output_shape[-2], model.output_shape[-1])))

model.add(Permute((2, 1))) 
model.add(Activation('softmax'))
model.compile(loss="categorical_crossentropy", optimizer= 'sgd' , metrics=['accuracy'])
'''
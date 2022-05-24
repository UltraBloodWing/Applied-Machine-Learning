import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
import random
import os


#--- SETTING PATHS ---

train_data = "train_data/"
val_data = "val_data/"
test_data = "test_data/"

#--- PREPROCESSING ---

#generate train images
train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale = 1/255.0,
                                                             rotation_range = 30,
                                                             zoom_range = 0.5,
                                                             horizontal_flip = True)

train_generator = train_datagen.flow_from_directory(val_data,
                                                    batch_size = 32,
                                                    class_mode = 'categorical',
                                                    target_size = (64, 64))

#generate validation images
val_datagen = keras.preprocessing.image.ImageDataGenerator(rescale = 1/255.0,
                                                           rotation_range = 30,
                                                           zoom_range = 0.5,
                                                           horizontal_flip = True)

val_generator = train_datagen.flow_from_directory(val_data,
                                                  batch_size = 32,
                                                  class_mode = 'categorical',
                                                  target_size = (64, 64))

#generate test images
test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale = 1/255.0,
                                                             rotation_range = 30,
                                                             zoom_range = 0.5,
                                                             horizontal_flip = True)

test_generator = train_datagen.flow_from_directory(test_data,
                                                    batch_size = 32,
                                                    class_mode = 'categorical' ,
                                                    target_size = (64, 64))

#--- BUILDING MODEL ---

#building a model by adding layers
model = Sequential([
                    #layer 1
                    Conv2D(64, (3, 3), activation='relu', input_shape = (64, 64, 3)),
                    BatchNormalization(),
                    MaxPooling2D(2, 2),
                    #layer 2
                    Conv2D(128, (5, 5), activation='relu'),
                    BatchNormalization(),
                    MaxPooling2D(2, 2),
                    #layer 3
                    Conv2D(512, (3, 3), activation='relu'),
                    BatchNormalization(),
                    MaxPooling2D(2, 2),
                    #layer 4
                    Conv2D(512, (3, 3), activation='relu'),
                    BatchNormalization(),
                    MaxPooling2D(2, 2),

                    Flatten(),
                    Dense(256, activation='relu'),
                    Dense(512, activation='relu'),
                    Dense(4000, activation='softmax')])

model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


#--- SETTING UP h5 FILE  and EARLY STOPPING ---

#setting up early stopping to prevent overfitting
earlystop = keras.callbacks.EarlyStopping(monitor ='val_loss',
                                          patience = 5,
                                          verbose = 1,
                                          restore_best_weights=True)

#save the best trained model
best_model_file_path = "model.h5"
best_model = keras.callbacks.ModelCheckpoint(best_model_file_path,
                                             monitor = 'val_loss',
                                             verbose = 1,
                                             save_best_only = True)

callbacks = [earlystop, best_model]

#--- TRAINING MODEL WITH TRAIN AND VAL ---

# fit model
history = model.fit(train_generator,
                    validation_data = val_generator,
                    batch_size=32,
                    epochs=70,
                    verbose=1,
                    callbacks = callbacks)


#--- PLOTTING GRAPHS TRAIN AND VAL ---

#Shows the graph for Train and Val Accuracy
acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc))

fig = plt.figure(figsize=(14,7))
plt.plot(epochs, acc, 'r', label="Training Accuracy")
plt.plot(epochs, val_acc, 'b', label="Validation Accuracy")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation accuracy')
plt.legend(['Training Accuracy','Validation Accuracy'],loc='lower right')
plt.show()

#Shows the graph for Train and Val Loss
fig2 = plt.figure(figsize=(14,7))
plt.plot(epochs, loss, 'r', label="Training Loss")
plt.plot(epochs, val_loss, 'b', label="Validation Loss")
plt.legend(['Training Loss','Validation Loss'],loc='upper right')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation loss')

#--- EVALUATION ---

#loading .h5 file 
model_file_path = "model.h5"
h5model = keras.models.load_model(model_file_path, compile = True)

#evaluate
results = model.evaluate(test_generator, batch_size=32)
print("Accuracy:", results[1])

#--- PREDICTION ---

# define function for testing the models
def model_tester(full_model):

    # This first part selects a random file from the validation directory
    types = random.choice(os.listdir(test_data))
    #print(types)
    file = random.choice(os.listdir('test_data/{0}'.format(types)))
    #print(file)

    random_path = 'test_data' + '/' + types + '/' + file
    #print(random_path)

    # We then create the list of labels 
    person_dict = train_generator.class_indices
    label_list = {v: k.lower().capitalize() for k, v in person_dict.items()}
    # We then select the image, preprocess and predict the values from full_model
    img_path = random_path
    img = image.load_img(img_path, target_size=(64,64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    #answer=full_model.predict(x)
    probability=round(np.max(full_model.predict(x)*100),2)
    
    # Prediction
    print ('This model suggests the image below is a: ',
           label_list[np.argmax(full_model.predict(x)*100)],
           ' with a probability of' ,probability,'%' ) 
    plt.imshow(img) 
   
    # Ground Truth
    print('____________________________________')
    print('The ground truth is: \t',types)
    print('____________________________________\n')


    #If we want to display the next two likely outcomes we can use:
    z = full_model.predict(x)*100
    temp = np.argpartition(z[0], -3)[-3:]
    #print(temp)
    #print(z[0][temp])
    temp = np.argsort(-z[0])[:3]

    print('The two next most likely choices are: \n', 
            '          ' , label_list[temp[1]], 'with probability', round(z[0][temp][1], 2),'% \n', 
            '          ' , label_list[temp[2]], 'with probability', round(z[0][temp][2], 2), '%' )
    print('____________________________________')
    
model_tester(full_model=model)












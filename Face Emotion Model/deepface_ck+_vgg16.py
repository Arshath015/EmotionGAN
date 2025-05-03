import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

# ✅ Dataset Path & Parameters
dataset_path = "Check/CK+_clahe"  
img_size = (224, 224)  
batch_size = 16  
num_classes = 7  

# ✅ Data Augmentation & Normalization
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,  
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    validation_split=0.2  
)

# ✅ Load Training & Validation Data (Now in RGB mode)
train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="training",
    color_mode="rgb"  # ✅ Changed from "grayscale" to "rgb"
)

val_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation",
    color_mode="rgb"  # ✅ Changed from "grayscale" to "rgb"
)

# ✅ Load Pretrained VGG16 Model (Without Top Layers)
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# ✅ Freeze Initial Layers
for layer in base_model.layers:
    layer.trainable = False

# ✅ Add Custom Fully Connected Layers
x = Flatten()(base_model.output)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.3)(x)
output_layer = Dense(num_classes, activation="softmax")(x)

# ✅ Define Model
model = Model(inputs=base_model.input, outputs=output_layer)

# ✅ Compile Model
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ✅ Callbacks for Training
checkpoint = ModelCheckpoint("Project/deepface/best_model.h5", monitor="val_accuracy", save_best_only=True, mode="max")
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-6)
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

# ✅ Train the Model
epochs = 5
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    callbacks=[checkpoint, reduce_lr, early_stop]
)

# ✅ Fine-Tune Last 10 Layers
for layer in base_model.layers[-10:]:  
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=0.00001),  
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ✅ Train Again (Fine-Tuning)
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5,
    callbacks=[checkpoint, reduce_lr, early_stop]
)

print("✅ Fine-tuning complete! Model saved as 'best_model.h5'.")

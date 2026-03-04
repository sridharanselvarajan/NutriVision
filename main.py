import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
import json
import matplotlib.pyplot as plt

# --- 1. DATA LOADING ---
# Load datasets from directories
train_ds = tf.keras.utils.image_dataset_from_directory(
    "data/train",
    image_size=(224, 224),
    batch_size=32
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "data/validation",
    image_size=(224, 224),
    batch_size=32
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    "data/test",
    image_size=(224, 224),
    batch_size=32
)

class_names = train_ds.class_names
print("Classes:", class_names)
num_classes = len(class_names)

# --- 2. DATASET OPTIMIZATION ---
# Create a data augmentation layer
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
])

# Preprocessing function for MobileNetV2
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

AUTOTUNE = tf.data.AUTOTUNE

# Apply augmentation and preprocessing
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE).cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE).cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE).cache().prefetch(buffer_size=AUTOTUNE)

# --- 3. MODEL BUILDING (TRANSFER LEARNING) ---
# Load the pre-trained MobileNetV2 model without its top classification layer
base_model = MobileNetV2(input_shape=(224, 224, 3),
                         include_top=False,
                         weights='imagenet')

# Freeze the convolutional base
base_model.trainable = False

# Build our custom classifier on top of the base model
inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x) # Add dropout for regularization
outputs = layers.Dense(num_classes, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)

# --- 4. MODEL COMPILATION ---
# Compile the model with an optimizer, loss function, and metrics
model.compile(
  optimizer='adam',
  loss='sparse_categorical_crossentropy',
  metrics=['accuracy']
)

model.summary()

# --- 5. MODEL TRAINING ---
print("\nStarting model training...")
initial_epochs = 10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=initial_epochs
)

# --- 6. FINE-TUNING THE MODEL ---
print("\nStarting fine-tuning...")
base_model.trainable = True # Unfreeze the base model

# Fine-tune from this layer onwards
fine_tune_at = 100

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Compile the model with a very low learning rate
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), # Low learning rate
    metrics=['accuracy']
)

model.summary()

fine_tune_epochs = 10
total_epochs = initial_epochs + fine_tune_epochs

history_fine = model.fit(
    train_ds,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1],
    validation_data=val_ds
)

print("Fine-tuning finished.")

# --- 7. VISUALIZE TRAINING RESULTS ---
# (Code to plot accuracy and loss will be added here in the next step if requested)

# --- 8. EVALUATE THE MODEL ---
print("\nEvaluating model on test data...")
loss, accuracy = model.evaluate(test_ds)
print(f"Test Accuracy: {accuracy:.4f}")

# --- 9. SAVE THE MODEL AND CLASS NAMES ---
model.save('food_classifier.keras')
with open('class_names.json', 'w') as f:
    json.dump(class_names, f)
print("\nModel and class names saved successfully!")

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Resizing, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback, TensorBoard
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B0, preprocess_input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import os
import gc

# EfficientNetV2B0 custom layers
@tf.keras.utils.register_keras_serializable()
class RepeatChannel(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.repeat(inputs, 3, axis=-1)
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], 3)

@tf.keras.utils.register_keras_serializable()
class PreprocessLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return preprocess_input(inputs)

# EfficientNetV2B0 model
df = '/kaggle/input/fer2013/fer2013.csv'
df = pd.read_csv(df)
final_V2B0_path = "/kaggle/working/final_v2b0_model.keras"
best_V2B0_path = "/kaggle/working/best_v2b0_model.keras"

def preprocess(df):
    X, y = [], []
    for px, lab in zip(df['pixels'], df['emotion']):
        arr = np.fromstring(px, dtype=np.uint8, sep=' ')
        if arr.size != 48*48:
            continue
        X.append(arr.reshape(48,48,1))
        y.append(lab)
    X = np.stack(X).astype('float32')
    y = tf.keras.utils.to_categorical(np.array(y), 7).astype('float32')
    return X, y

train_X, train_y = preprocess(df[df['Usage']=='Training'])
val_X, val_y = preprocess(df[df['Usage']=='PrivateTest'])
test_X, test_y = preprocess(df[df['Usage']=='PublicTest'])

def augment(img, label):
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 0.2)
    img = tf.image.random_contrast(img, 0.8, 1.2)
    img = img * 255.0
    return img, label

BATCH = 64
AUTOTUNE = tf.data.AUTOTUNE

train_ds = (tf.data.Dataset.from_tensor_slices((train_X, train_y))
            .shuffle(10000)
            .map(augment, num_parallel_calls=AUTOTUNE)
            .batch(BATCH)
            .prefetch(AUTOTUNE))

val_ds = (tf.data.Dataset.from_tensor_slices((val_X, val_y))
          .batch(BATCH)
          .prefetch(AUTOTUNE))

test_ds = (tf.data.Dataset.from_tensor_slices((test_X, test_y))
           .batch(BATCH)
           .prefetch(AUTOTUNE))

def build_model():
    inputs = Input(shape=(48,48,1))
    x = Resizing(224,224)(inputs)
    x = RepeatChannel()(x)
    x = PreprocessLayer()(x)
    base = EfficientNetV2B0(weights='imagenet', include_top=False)
    base.trainable = False
    x = base(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(192, activation='swish')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(7, activation='softmax')(x)
    return Model(inputs, outputs), base

model, base = build_model()

from collections import Counter
counts = Counter(np.argmax(train_y, axis=1))
max_count = max(counts.values())
class_weight = {c: (max_count/counts.get(c, max_count))**0.5 for c in range(7)}
print("Class weights:", class_weight)

class PrintEpoch(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch+1:02d} | Loss: {logs['loss']:.4f} | Acc: {logs['accuracy']:.4f} | "
              f"Val Loss: {logs['val_loss']:.4f} | Val Acc: {logs['val_accuracy']:.4f}")

callbacks = [
    PrintEpoch(),
    EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
    ModelCheckpoint(best_V2B0_path, monitor='val_accuracy', save_best_only=True),
    TensorBoard(log_dir='./logs')
]

loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)
optimizer = Adam(learning_rate=5e-5)
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

print("\nStarting Warm-up Training...")
history_warm = model.fit(train_ds, epochs=5, validation_data=val_ds, callbacks=callbacks)

for layer in base.layers[-60:]:
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True

optimizer_ft = Adam(learning_rate=1e-4)
model.compile(optimizer=optimizer_ft, loss=loss_fn, metrics=['accuracy'])

print("\nStarting Fine-tuning...")
history_fine = model.fit(train_ds, epochs=25, validation_data=val_ds,
                         callbacks=callbacks, class_weight=class_weight)

print("\nEvaluating on test set...")
test_loss, test_acc = model.evaluate(test_ds)
print(f"\nTest Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history_warm.history['accuracy'] + history_fine.history['accuracy'], label='Train Acc')
plt.plot(history_warm.history['val_accuracy'] + history_fine.history['val_accuracy'], label='Val Acc')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

y_pred = np.argmax(model.predict(test_ds), axis=1)
y_true = np.argmax(test_y, axis=1)

class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4,        
    zero_division=0   
)
print('\nClassification Report:\n')
print(report)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.show()

model.save(final_V2B0_path)
print("Model saved successfully!")

del train_X, train_y, val_X, val_y, test_X, test_y, df
gc.collect()

# CNN Model
best_CNN_path  = "/kaggle/working/best_CNN_model.keras"
final_CNN_path = "/kaggle/working/final_CNN_model.keras"

df = pd.read_csv('/kaggle/input/fer2013/fer2013.csv')
X = np.stack(df['pixels'].apply(lambda x: np.fromstring(x, dtype='float32', sep=' '))).reshape(-1,48,48,1)/255.0
y = pd.get_dummies(df['emotion']).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val   = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

model = Sequential([
    Conv2D(64,  (3,3), activation='relu', input_shape=(48,48,1)), BatchNormalization(), MaxPooling2D(), Dropout(0.3),
    Conv2D(128, (3,3), activation='relu'),                       BatchNormalization(), MaxPooling2D(), Dropout(0.4),
    Conv2D(256, (3,3), activation='relu'),                       BatchNormalization(), MaxPooling2D(), Dropout(0.5),
    Flatten(),
    Dense(256, activation='relu', kernel_regularizer=l2(0.01)),  BatchNormalization(), Dropout(0.6),
    Dense(7, activation='softmax')
])

model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint   = ModelCheckpoint(best_CNN_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
early_stop   = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr    = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    batch_size=64, epochs=30,
                    callbacks=[checkpoint, early_stop, reduce_lr])

model.save(final_CNN_path)
print("✅ Saved:", final_CNN_path)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

print("\nClassification Report:")
print(classification_report(y_true_labels, y_pred_labels))

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

cm = confusion_matrix(y_true_labels, y_pred_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral'],
            yticklabels=['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.show()

# Vision Transformer (ViT) Model
image_size = 48
patch_size = 6
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_layers = 8

best_Vit_path  = "/kaggle/working/best_Vit_model.keras"
final_Vit_path = "/kaggle/working/final_Vit_model.keras"

df = '/kaggle/input/fer2013/fer2013.csv'
data = pd.read_csv(df)

pixels = data['pixels'].apply(lambda x: np.array(x.split(), dtype='float32'))
X = np.stack(pixels.values, axis=0)
X = X.reshape(-1, 48, 48, 1)
X = X / 255.0

y = pd.get_dummies(data['emotion']).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)  # 60-20-20

class Patches(tf.keras.layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

def create_vit_model():
    inputs = tf.keras.layers.Input(shape=(48, 48, 1))
    patches = Patches(patch_size)(inputs)
    projected_patches = tf.keras.layers.Dense(projection_dim)(patches)
    positions = tf.range(start=0, limit=num_patches, delta=1)
    positional_embedding = tf.keras.layers.Embedding(
        input_dim=num_patches, output_dim=projection_dim
    )(positions)
    encoded = projected_patches + positional_embedding
    for _ in range(transformer_layers):
        x1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded)
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim//num_heads
        )(x1, x1)
        x2 = tf.keras.layers.Add()([attention_output, encoded])
        x3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = tf.keras.layers.Dense(projection_dim * 2, activation="gelu")(x3)
        x3 = tf.keras.layers.Dense(projection_dim)(x3)
        encoded = tf.keras.layers.Add()([x3, x2])
    representation = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded)
    representation = tf.keras.layers.Flatten()(representation)
    representation = tf.keras.layers.Dropout(0.5)(representation)
    outputs = tf.keras.layers.Dense(7, activation="softmax")(representation)
    return Model(inputs=inputs, outputs=outputs)

vit_model = create_vit_model()
vit_model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

vit_model.summary()

history_vit = vit_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=32,
    epochs=50,
    callbacks=[
        EarlyStopping(patience=10, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=5),
        ModelCheckpoint(
            filepath=best_Vit_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
    ]
)

y_pred = vit_model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

print("\nClassification Report:")
print(classification_report(y_true_labels, y_pred_labels))

vit_model.save(final_Vit_path)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history_vit.history['accuracy'], label='Train Acc')
plt.plot(history_vit.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_vit.history['loss'], label='Train Loss')
plt.plot(history_vit.history['val_loss'], label='Val Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

cm = confusion_matrix(y_true_labels, y_pred_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral'],
            yticklabels=['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.show()

# Gradio UI
import gradio as gr
from PIL import Image

theme = gr.themes.Default(
    text_size=gr.themes.sizes.text_lg,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
)

@tf.keras.utils.register_keras_serializable(package='Custom')
class PreprocessLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, inputs):
        return tf.keras.applications.efficientnet.preprocess_input(inputs)
    def get_config(self):
        return super().get_config()

@tf.keras.utils.register_keras_serializable()
class RepeatChannel(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, inputs):
        return tf.repeat(inputs, 3, axis=-1)
    def get_config(self):
        return super().get_config()

@tf.keras.utils.register_keras_serializable()
class Patches(tf.keras.layers.Layer):
    def __init__(self, patch_size=6, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID")
        return tf.reshape(patches, [batch_size, -1, patches.shape[-1]])
    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config

emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

custom_objects = {
    "RepeatChannel": RepeatChannel,
    "Patches": Patches,
    "PreprocessLayer": PreprocessLayer,
    "Custom>RepeatChannel": RepeatChannel,
    "Custom>Patches": Patches,
    "Custom>PreprocessLayer": PreprocessLayer
}

def load_single_model(path):
    try:
        return tf.keras.models.load_model(
            path,
            custom_objects=custom_objects,
            compile=False,
            safe_mode=False
        )
    except Exception as e:
        print(f"❌ error loading «{os.path.basename(path)}»: {e}")
        return None

def load_models():
    models = {}
    try:
        models["EfficientNetV2B0"] = tf.keras.models.load_model(
            "/kaggle/input/v2b0keras/best_v2b0_model.keras",
            custom_objects=custom_objects,
            compile=False,
            safe_mode=False
        )
    except Exception as e:
        print(f" final error: {str(e)}")
        print("\nother solution with weights")
        models["EfficientNetV2B0"] = build_effnet_from_weights()
    models["ViT"] = tf.keras.models.load_model(
        "/kaggle/input/vitkeras/best_Vit_model.keras",
        custom_objects={"Patches": Patches},
        compile=False
    )
    models["CNN"] = tf.keras.models.load_model(
        "/kaggle/input/cnnkeras/best_CNN_model.keras",
        compile=False
    )
    return models

print("in progress loading...")
models = load_models()
available_models = {name: model for name, model in models.items() if model is not None}
print("loaded models:", list(available_models.keys()))

if not available_models:
    raise ValueError("no model is loaded!")

def predict_emotion(img, model_name):
    model = available_models[model_name]
    try:
        img = img.convert("L").resize((48, 48))
        arr = np.array(img).reshape(1, 48, 48, 1).astype("float32") / 255.0
        pred = model.predict(arr)
        return f"‌predict: {emotion_labels[np.argmax(pred)]} (model: {model_name})"
    except Exception as e:
        return f"prediction error  : {str(e)}"

iface = gr.Interface(
    fn=predict_emotion,
    inputs=[
        gr.Image(type="pil", label="Upload an Image"),
        gr.Radio(list(available_models.keys()), label="Choose a Model")
    ],
    outputs=gr.Label(label="Result"),
    title="Facial Emotion Recognition",
    description="Upload an image and select a model to predict the emotion.",
)
iface.launch(share=True)
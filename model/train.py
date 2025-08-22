import argparse
import json
import os
import tensorflow as tf
from tensorflow.keras import layers, callbacks

def build_model(input_shape=(150,150,3), n_classes=4):
    model = tf.keras.Sequential([
        layers.Input(shape=input_shape),
        # augmentation is applied as separate preprocessing layer below
        layers.Rescaling(1./255),


        layers.Conv2D(32, (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.3),


        layers.Conv2D(64, (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.3),


        layers.Conv2D(128, (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.3),


        layers.Conv2D(128, (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.3),


        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(n_classes, activation='softmax')
        ])
    return model

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, default='data/', help='Directory with class subfolders')
    p.add_argument('--image_size', type=int, default=150)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--model_output', type=str, default='model/best_model.h5')
    p.add_argument('--labels_output', type=str, default='model/labels.json')
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.model_output) or '.', exist_ok=True)


    img_size = (args.image_size, args.image_size)
    batch_size = args.batch_size
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        args.data_dir,
        validation_split=0.1,
        subset='training',
        seed=101,
        image_size=img_size,
        batch_size=batch_size
)


    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        args.data_dir,
        validation_split=0.1,
        subset='validation',
        seed=101,
        image_size=img_size,
        batch_size=batch_size
)
    class_names = train_ds.class_names
    n_classes = len(class_names)
    print('Classes:', class_names)

    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(0.06),
        layers.RandomZoom(0.06),
        ])
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


    model = build_model(input_shape=(args.image_size, args.image_size, 3), n_classes=n_classes)
    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    cb_list = [
        callbacks.ModelCheckpoint(args.model_output, save_best_only=True, monitor='val_accuracy', mode='max'),
        callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4)
]
    history = model.fit(
        train_ds.map(lambda x, y: (data_augmentation(x, training=True), y)),
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=cb_list
)
    with open(args.labels_output, 'w') as f:
        json.dump(class_names, f)
    print('Training finished. Model saved to', args.model_output)

if __name__ == '__main__':
    main()    

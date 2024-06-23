import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Dropout, GlobalAveragePooling2D, Input, Lambda
from tensorflow.keras import applications
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.backend import resize_images
from utitlities.spot_10_reader import SPOT10Loader

def build_model(model_name=None, num_classes=10, weights='imagenet', input_shape=(32, 32, 1)):
    input_layer = Input(shape=input_shape)
    # Convert the grayscale input to 3 channels
    x = Conv2D(3, (1, 1), padding='same')(input_layer)
    # Resize the image to the required size for the pre-trained models
    x = Lambda(lambda image: resize_images(image, height_factor=3, width_factor=3, data_format='channels_last'))(x)

    base_model = None

    if model_name == "Xception":
        base_model = applications.Xception(input_shape=(x.shape[1], x.shape[2], x.shape[3]), include_top=False, weights=weights)
    elif model_name == "ResNet50":
        base_model = applications.ResNet50(input_shape=(x.shape[1], x.shape[2], x.shape[3]), include_top=False, weights=weights)
    elif model_name == "InceptionV3":
        base_model = applications.InceptionV3(input_shape=(x.shape[1], x.shape[2], x.shape[3]), include_top=False, weights=weights)
    elif model_name == "InceptionResNetV2":
        base_model = applications.InceptionResNetV2(input_shape=(x.shape[1], x.shape[2], x.shape[3]), include_top=False, weights=weights)
    elif model_name == "MobileNet":
        base_model = applications.MobileNet(input_shape=(x.shape[1], x.shape[2], x.shape[3]), include_top=False, weights=weights)
    elif model_name == "MobileNetV2":
        base_model = applications.MobileNetV2(input_shape=(x.shape[1], x.shape[2], x.shape[3]), include_top=False, weights=weights)
    elif model_name == "DenseNet121":
        base_model = applications.DenseNet121(input_shape=(x.shape[1], x.shape[2], x.shape[3]), include_top=False, weights=weights)
    elif model_name == "NASNetMobile":
        base_model = applications.NASNetMobile(input_shape=(x.shape[1], x.shape[2], x.shape[3]), include_top=False, weights=weights)
    elif model_name == "EfficientNetB0":
        base_model = applications.EfficientNetB0(input_shape=(x.shape[1], x.shape[2], x.shape[3]), include_top=False, weights=weights)
    elif model_name == "EfficientNetB1":
        base_model = applications.EfficientNetB1(input_shape=(x.shape[1], x.shape[2], x.shape[3]), include_top=False, weights=weights)
    elif model_name == "EfficientNetB2":
        base_model = applications.EfficientNetB2(input_shape=(x.shape[1], x.shape[2], x.shape[3]), include_top=False, weights=weights)
    elif model_name == "EfficientNetB3":
        base_model = applications.EfficientNetB3(input_shape=(x.shape[1], x.shape[2], x.shape[3]), include_top=False, weights=weights)
    else:
        raise ValueError("Please specify a valid model!")

    x = base_model(x, training=False)

    # Add global average pooling layer
    x = GlobalAveragePooling2D()(x)
    # Add a fully-connected layer and a softmax layer with num_classes outputs
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=predictions)
    return model

# Load data
X_train, y_train = SPOT10Loader.get_data(dataset_dir="dataset", kind="train")
X_test, y_test = SPOT10Loader.get_data(dataset_dir="dataset", kind="test")

# Normalize the data
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Reshape the data to include the channel dimension
X_train = tf.expand_dims(X_train, axis=-1)
X_test = tf.expand_dims(X_test, axis=-1)

# One-hot encode the labels
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)



# Build and compile model
model = build_model(model_name="DenseNet121")
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
]

# Train model
history = model.fit(
    X_train, y_train,
    batch_size=32,
    validation_data=(X_test, y_test),
    epochs=100,
    callbacks=callbacks
)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")
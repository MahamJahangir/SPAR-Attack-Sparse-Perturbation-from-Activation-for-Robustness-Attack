
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

def build_and_train_model(X_train, y_train, X_test, y_test):
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        Conv2D(32, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=128, epochs=30, validation_data=(X_test, y_test))
    model.save_weights("models/ksvdr-mnist-model.h5")
    return model

def load_model_weights(model):
    model.load_weights("models/ksvdr-mnist-model.h5")
    return model

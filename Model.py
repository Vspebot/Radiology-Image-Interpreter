import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Directory setup
main_dir = "C:\\Users\\Admin\\Desktop\\Python Mega Course Build 10 Real World Applications\\pythonProject3\\Chext-X-ray-Images-Data-Set\\DataSet\\Data"
train_dir = os.path.join(main_dir, "train")
test_dir = os.path.join(main_dir, "test")

print("Train directory:", train_dir)
print("Test directory:", test_dir)


# Data visualization (sample images)
def visualize_images():
    rows = 4
    columns = 4

    fig = plt.gcf()
    fig.set_size_inches(12, 12)

    train_covid_dir = os.path.join(train_dir, "COVID19")
    train_normal_dir = os.path.join(train_dir, "NORMAL")

    print("Train COVID directory:", train_covid_dir)
    print("Train NORMAL directory:", train_normal_dir)

    train_covid_names = os.listdir(train_covid_dir)[:8]
    train_normal_names = os.listdir(train_normal_dir)[:8]

    covid_img = [os.path.join(train_covid_dir, filename) for filename in train_covid_names]
    normal_img = [os.path.join(train_normal_dir, filename) for filename in train_normal_names]

    merged_img = covid_img + normal_img

    for i, img_path in enumerate(merged_img):
        title = img_path.split(os.sep)[-1]
        plot = plt.subplot(rows, columns, i + 1)
        plot.axis("Off")
        img = plt.imread(img_path)
        plot.set_title(title, fontsize=11)
        plt.imshow(img, cmap="gray")

    plt.show()


# Data preprocessing and augmentation
def setup_data_generators():
    dgen_train = ImageDataGenerator(rescale=1. / 255,
                                    validation_split=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)

    dgen_validation = ImageDataGenerator(rescale=1. / 255)

    dgen_test = ImageDataGenerator(rescale=1. / 255)

    train_generator = dgen_train.flow_from_directory(train_dir,
                                                     target_size=(150, 150),
                                                     subset='training',
                                                     batch_size=32,
                                                     class_mode='binary')

    validation_generator = dgen_validation.flow_from_directory(train_dir,
                                                               target_size=(150, 150),
                                                               subset="validation",
                                                               batch_size=32,
                                                               class_mode="binary")

    test_generator = dgen_test.flow_from_directory(test_dir,
                                                   target_size=(150, 150),
                                                   batch_size=32,
                                                   class_mode="binary")

    return train_generator, validation_generator, test_generator


# Building the CNN model
def build_model(input_shape):
    model = Sequential()

    model.add(Conv2D(32, (5, 5), padding="same", activation="relu", input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(64, (5, 5), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))

    return model


# Compiling and training the model
def train_model(model, train_generator, validation_generator):
    model.compile(Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(train_generator,
                        epochs=35,
                        validation_data=validation_generator)

    return history


# Evaluation and plotting
def evaluate_model(model, test_generator):
    test_loss, test_acc = model.evaluate(test_generator)
    print("Test Set Loss : ", test_loss)
    print("Test Set Accuracy : ", test_acc)

    # Fetching history from the model
    history = model.history.history

    plt.figure(figsize=(12, 6))

    # Plotting training and validation loss
    if 'loss' in history:
        plt.subplot(1, 2, 1)
        plt.plot(history['loss'])
        if 'val_loss' in history:
            plt.plot(history['val_loss'])
        plt.legend(['Training', 'Validation'])
        plt.title("Training and validation losses")
        plt.xlabel('Epoch')

    # Plotting training and validation accuracy
    if 'accuracy' in history:
        plt.subplot(1, 2, 2)
        plt.plot(history['accuracy'])
        if 'val_accuracy' in history:
            plt.plot(history['val_accuracy'])
        plt.legend(['Training', 'Validation'])
        plt.title("Training and validation accuracy")
        plt.xlabel('Epoch')

    plt.tight_layout()
    plt.show()

    # Save the model
    model.save("model.h5")
    print("Model saved as 'model.h5'")


# Main execution flow
if __name__ == "__main__":
    visualize_images()

    train_gen, val_gen, test_gen = setup_data_generators()

    input_shape = train_gen.image_shape
    model = build_model(input_shape)

    history = train_model(model, train_gen, val_gen)

    evaluate_model(model, test_gen)

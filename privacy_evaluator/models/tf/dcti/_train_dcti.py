import tensorflow as tf

from privacy_evaluator.models.tf.dcti.dcti import DCTI


def augment(x, y):
    x = tf.image.resize_with_crop_or_pad(x, 32 + 8, 32 + 8)
    x = tf.image.random_crop(x, [32, 32, 3])
    x = tf.image.random_flip_left_right(x)
    return x, y


def normalize(x, y):
    x = tf.image.per_image_standardization(x)
    return x, y


def scale(x):
    return x / 255


def main():
    (x, y), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Scale to [0, 1]
    x = scale(x)
    x_test = scale(x_test)

    train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    train_dataset = (
        train_dataset.map(augment)
        .map(normalize)
        .shuffle(buffer_size=len(train_dataset))
        .batch(128)
    )
    test_dataset = test_dataset.map(normalize).batch(128)

    model = DCTI()
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        metrics=["accuracy"],
    )

    model.fit(
        train_dataset, epochs=100, validation_data=test_dataset, validation_freq=1
    )

    model.save("./model")


if __name__ == "__main__":
    main()

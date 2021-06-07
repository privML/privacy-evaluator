import tensorflow as tf
from tensorflow.keras.layers import Conv1D, BatchNormalization, ReLU, MaxPool1D, Dense, Dropout, Input, Flatten



class ConvNetMetaClassifier(tf.keras.Model):
    def __init__(self, inputs: Input, num_classes: int=2):
        super().__init__()

        self.model = self.DeepMetaClassifier(inputs, num_classes)



    def ConvBlock(inputs, filters, pool=True):

        x = Conv1D(filters, 5)(inputs)
        x = ReLU()(x)
        x = BatchNormalization()(x)
        if pool:
            x = MaxPool1D(pool_size=2)(x)

        return x

    def DenseBlock(inputs, units):

        x = Dense(units)(inputs)
        x = Dropout(rate=0.5)(x)
        x = ReLU()(x)
        x = BatchNormalization()(x)

        return x

    def DeepMetaClassifier(self, inputs: Input, num_classes: int=2):
    # This is the Deep Meta-Classifier (DMC) taken from https://arxiv.org/pdf/2002.05688.pdf
    # I think there is one layer more here than in the paper.

        x = inputs
        for filters in [8, 16, 32, 64]:
            x = self.ConvBlock(x, filters)

        for i in range(5):
            x = self.ConvBlock(x, 128)

        x = self.ConvBlock(x, 128, pool=False)

        for i in range(5):
            x = self.ConvBlock(x, 256, pool=(i % 2 == 0))

        x = Flatten()(x)

        for i in range(4):
            x = self.DenseBlock(x, 1024)

        x = self.DenseBlock(x, 64)

        x = Dense(num_classes)(x)

        model = tf.keras.Model(inputs=inputs, outputs=x, name="DeepMetaClassifier")
        return model

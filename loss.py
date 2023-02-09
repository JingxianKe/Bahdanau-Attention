# import the necessary packages
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.losses import Loss
import tensorflow as tf
class MaskedLoss(Loss):
        def __init__(self):
                # initialize the name of the loss and the loss function
                self.name = "masked_loss"
                self.loss = SparseCategoricalCrossentropy(from_logits=True,
                        reduction="none")
        def __call__(self, yTrue, yPred):
                # calculate the loss for each item in the batch
                loss = self.loss(yTrue, yPred)
                # mask off the losses on padding
                mask = tf.cast(yTrue != 0, tf.float32)
                loss *= mask
                # return the total loss
                return tf.reduce_sum(loss)
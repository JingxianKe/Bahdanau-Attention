# import the necessary packages
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
import tensorflow as tf
import numpy as np
class WarmUpCosine(LearningRateSchedule):
        def __init__(self, lrStart, lrMax, warmupSteps, totalSteps):
                super().__init__()
                self.lrStart = lrStart
                self.lrMax = lrMax
                self.warmupSteps = warmupSteps
                self.totalSteps = totalSteps
                self.pi = tf.constant(np.pi)

        def __call__(self, step):
                # check whether the total number of steps is larger than the
                # warmup steps. If not, then throw a value error
                if self.totalSteps < self.warmupSteps:
                        raise ValueError(
                                f"Total number of steps {self.totalSteps} must be"
                                + f"larger or equal to warmup steps {self.warmupSteps}."
                        )
                # a graph that increases to 1 from the initial step to the
                # warmup step, later decays to -1 at the final step mark
                cosAnnealedLr = tf.cos(
                        self.pi
                        * (tf.cast(step, tf.float32) - self.warmupSteps)
                        / tf.cast(self.totalSteps - self.warmupSteps, tf.float32)
                )
                # shift the learning rate and scale it
                learningRate = 0.5 * self.lrMax * (1 + cosAnnealedLr)
                
                # check whether warmup steps is more than 0.
                if self.warmupSteps > 0:
                        # throw a value error is max lr is smaller than start lr
                        if self.lrMax < self.lrStart:
                                raise ValueError(
                                        f"lr_start {self.lrStart} must be smaller or"
                                        + f"equal to lr_max {self.lrMax}."
                                )
                        # calculate the slope of the warmup line and build the
                        # warmup rate
                        slope = (self.lrMax - self.lrStart) / self.warmupSteps
                        warmupRate = slope * tf.cast(step, tf.float32) + self.lrStart
                        # when the current step is lesser that warmup steps, get
                        # the line graph, when the current step is greater than
                        # the warmup steps, get the scaled cos graph.
                        learning_rate = tf.where(
                                step < self.warmupSteps, warmupRate, learningRate
                        )
                # return the lr schedule
                return tf.where(
                        step > self.totalSteps, 0.0, learningRate,
                        name="learning_rate",
                )
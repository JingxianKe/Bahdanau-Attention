# import the necessary packages
from tensorflow.keras.layers import AdditiveAttention
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GRU
from tensorflow.keras import Sequential
import tensorflow as tf

class Encoder(Layer):
        def __init__(self, sourceVocabSize, embeddingDim, encUnits,
                 **kwargs):
                super().__init__(**kwargs)
                # initialize the source vocab size, embedding dimensions, and
                # the encoder units
                self.sourceVocabSize = sourceVocabSize
                self.embeddingDim = embeddingDim
                self.encUnits = encUnits

        def build(self, inputShape):
                # the embedding layer converts token IDs to embedding vectors
                self.embedding = Embedding(
                        input_dim=self.sourceVocabSize,
                        output_dim=self.embeddingDim,
                        mask_zero=True,
                )

                # the GRU layer processes the embedding vectors sequentially
                self.gru = Bidirectional(
                        GRU(
                                units=self.encUnits,
                                # return the sequence and the state
                                return_sequences=True,
                                return_state=True,
                                recurrent_initializer="glorot_uniform",
                        )
                )

        def get_config(self):
                # return the configuration of the encoder layer
                return {
                        "inputVocabSize": self.inputVocabSize,
                        "embeddingDim": self.embeddingDim,
                        "encUnits": self.encUnits,
                }

        def call(self, sourceTokens, state=None):
                # pass the source tokens through the embedding layer to get
                # source vectors
                sourceVectors = self.embedding(sourceTokens)

                # create the masks for the source tokens
                sourceMask = self.embedding.compute_mask(sourceTokens)

                # pass the source vectors through the GRU layer
                (encOutput, encFwdState, encBckState) = self.gru(
                        inputs=sourceVectors,
                        initial_state=state,
                        mask=sourceMask
                )

                # return the encoder output, encoder state, and the
                # source mask
                return (encOutput, encFwdState, encBckState, sourceMask)

class BahdanauAttention(Layer):
        def __init__(self, attnUnits, **kwargs):
                super().__init__(**kwargs)
                # initialize the attention units
                self.attnUnits = attnUnits

        def build(self, inputShape):
                # the dense layers projects the query and the value
                self.denseEncoderAnnotation = Dense(
                        units=self.attnUnits,
                        use_bias=False,
                )
                self.denseDecoderAnnotation = Dense(
                        units=self.attnUnits,
                        use_bias=False,
                )

                # build the additive attention layer
                self.attention = AdditiveAttention()

        def get_config(self):
                # return the configuration of the layer
                return {
                        "attnUnits": self.attnUnits,
                }

        def call(self, hiddenStateEnc, hiddenStateDec, mask):
                # grab the source and target mask
                sourceMask = mask[0]
                targetMask = mask[1]

                # pass the query and value through the dense layer
                encoderAnnotation = self.denseEncoderAnnotation(hiddenStateEnc)
                decoderAnnotation = self.denseDecoderAnnotation(hiddenStateDec)

                # apply attention to align the representations
                (contextVector, attentionWeights) = self.attention(
                        inputs=[decoderAnnotation, hiddenStateEnc, encoderAnnotation],
                        mask=[targetMask, sourceMask],
                        return_attention_scores=True
                )

                # return the context vector and the attention weights
                return (contextVector, attentionWeights)

class Decoder(Layer):
        def __init__(self, targetVocabSize, embeddingDim, decUnits, **kwargs):
                super().__init__(**kwargs)
                # initialize the target vocab size, embedding dimension, and
                # the decoder units
                self.targetVocabSize = targetVocabSize
                self.embeddingDim = embeddingDim
                self.decUnits = decUnits

        def get_config(self):
                # return the configuration of the layer
                return {
                        "targetVocabSize": self.targetVocabSize,
                        "embeddingDim": self.embeddingDim,
                        "decUnits": self.decUnits,
                }

        def build(self, inputShape):
                # build the embedding layer which converts token IDs to
                # embedding vectors
                self.embedding = Embedding(
                    input_dim=self.targetVocabSize,
                    output_dim=self.embeddingDim,
                    mask_zero=True,
                )

                # build the GRU layer which processes the embedding vectors
                # in a sequential manner
                self.gru = GRU(
                        units=self.decUnits,
                        return_sequences=True,
                        return_state=True,
                        recurrent_initializer="glorot_uniform"
                )

                # build the attention layer
                self.attention = BahdanauAttention(self.decUnits)

                # build the final output layer
                self.fwdNeuralNet = Sequential([
                        Dense(
                                units=self.decUnits,
                                activation="tanh",
                                use_bias=False,
                        ),
                        Dense(
                                units=self.targetVocabSize,
                        ),
                ])

        def call(self, inputs, state=None):
                # grab the target tokens, encoder output, and source mask
                targetTokens = inputs[0]
                encOutput = inputs[1]
                sourceMask = inputs[2]

                # get the target vectors by passing the target tokens through
                # the embedding layer and create the target masks
                targetVectors = self.embedding(targetTokens)
                targetMask = self.embedding.compute_mask(targetTokens)

                # process one step with the GRU
                (decOutput, decState) = self.gru(inputs=targetVectors,
                                                 initial_state=state, mask=targetMask)

                # use the GRU output as the query for the attention over the
                # encoder output
                (contextVector, attentionWeights) = self.attention(
                        hiddenStateEnc=encOutput,
                        hiddenStateDec=decOutput,
                        mask=[sourceMask, targetMask],
                )

                # concatenate the context vector and output of GRU layer
                contextAndGruOutput = tf.concat(
                        [contextVector, decOutput], axis=-1)

                # generate final logit predictions
                logits = self.fwdNeuralNet(contextAndGruOutput)

                # return the predicted logits, attention weights, and the
                # decoder state
                return (logits, attentionWeights, decState)

import tensorflow as tf
import numpy as np

class ownS2SWithAttention(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super(ownS2SWithAttention, self).__init__(*args, **kwargs)
        self.encoder_input_layer = tf.keras.layers.Input((None, None), batch_size=None)
        self.encoder_embed_layer = tf.keras.layers.Embedding(input_dim=None, output_dim=None)
        self.encoder_masking_layer = tf.keras.layers.Masking(mask_value=0.0)
        self.encoder_LSTM_layer = tf.keras.layers.LSTM(None, return_sequences=True, return_state=True, dropout=0.3)
        self.encoder_state_dense_layer = tf.keras.layers.Dense(None, activation="elu")
        self.last_state_dense_layer = tf.keras.layers.Dense(None, activation="elu")

        self.encoder_tanh_layer = tf.keras.layers.Dense(None, activation="tanh")
        self.encoder_softmax_layer = tf.keras.layers.Dense(None, activation="softmax")

        self.decoder_input_layer = tf.keras.layers.Input((None, None), batch_size=None)
        self.decoder_embed_layer = tf.keras.layers.Embedding(input_dim=None, output_dim=None)

        self.decoder_tanh_layer = tf.keras.layers.Dense(None, activation="tanh")
        self.decoder_softmax_layer = tf.keras.layers.Dense(None, activation="softmax")

    def call(self, inputs, training=None, mask=None):
        prediction = [[0.]*100]  # output_dim
        prediction[0][1] = 1.  # sos token
        prediction[0] = tf.convert_to_tensor(prediction[0])

        decoder_embed = None
        if training:
            inputs, decoder_inputs = inputs
            decoder_inputs = self.decoder_input_layer(decoder_inputs)
            decoder_embed = self.decoder_embed_layer(decoder_inputs)

        inputs = self.input_layer(inputs)
        embed = self.embed_layer(inputs)
        mask = self.encoder_masking_layer(embed)
        encoder_states, state_h, state_c = self.encoder_LSTM_layer(mask)

        dense_encoder_states = self.encoder_state_dense_layer(encoder_states)
        last_state = self.last_state_dense_layer(state_h)

        while np.argmax(prediction[-1]) != 0:  # <eos>
            attention_score = self.encoder_tanh_layer(dense_encoder_states + last_state)
            attention_weight = self.encoder_softmax_layer(attention_score)
            context_vector = np.sum(encoder_states * attention_weight, axis=1)
            if training:
                concat_vector = tf.concat([context_vector, decoder_embed[:, :, len(prediction)]], -1)
            else:
                if len(prediction) == 1:
                    concat_vector = tf.concat([context_vector, last_state], -1)  # concat with <sos> (어떻게 연결하는지 알아봐야 함)
                else:
                    concat_vector = tf.concat([context_vector, last_state], -1)

            decoder_tanh = self.decoder_tanh_layer(concat_vector)
            last_state = self.decoder_softmax_layer(decoder_tanh)  # last_state가 바뀐 후에도 동작?
            prediction.append(last_state)
        return prediction

    def get_config(self):
        return self.config

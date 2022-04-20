import tensorflow as tf
from tensorflow.keras.layers import *

def sample_S2S():
    src_size = 300
    tar_size = 250
    embedding_size = 128

    encoder_inputs = tf.keras.layers.Input(shape=(None, src_size))
    encoder_embedding = tf.keras.layers.Embedding(src_size, embedding_size)(encoder_inputs)
    encoder_masking = tf.keras.layers.Masking(mask_value=0.0)(encoder_embedding)
    encoder_lstm = tf.keras.layers.LSTM(256, return_state=True)
    encoder_output, state_h, state_c = encoder_lstm(encoder_masking)
    encoder_states = [state_h, state_c]

    decoder_inputs = tf.keras.layers.Input(shape=(None, tar_size))
    decoder_embedding = tf.keras.layers.Embedding(tar_size, embedding_size)(decoder_inputs)
    decoder_masking = tf.keras.layers.Masking(mask_value=0.0)(decoder_embedding)
    decoder_lstm = tf.keras.layers.LSTM(256, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_masking, initial_state=encoder_states)

    decoder_softmax = tf.keras.layers.Dense(tar_size, activation="softmax")
    decoder_outputs = decoder_softmax(decoder_outputs)

    # train_model
    model = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer="adam", loss="categorical_crossentropy")
    # model.fit()

    decoder_state_input_h = tf.keras.layers.Input(shape=(embedding_size,))
    decoder_state_input_c = tf.keras.layers.Input(shape=(embedding_size,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_inputs = tf.keras.layers.Input(shape=(None, tar_size))
    decoder_embedding = tf.keras.layers.Embedding(tar_size, embedding_size)(decoder_inputs)

    decoder_outputs, state_h, state_c = decoder_lstm(decoder_embedding, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_softmax(decoder_outputs)
    # test_model
    encoder_model = tf.keras.Model(encoder_inputs, encoder_states)
    decoder_model = tf.keras.Model(inputs=[decoder_inputs] + decoder_states_inputs, outputs=[decoder_outputs] + decoder_states)
    # 이 후 입력을 encoder model에 넣고, |<sos>+인코더 cell/hidden_state상태| > |현재 토큰+반환된 디코더 상태|를 반환된 토큰이 <eos>일떄까지 반복해 decoder model에 넣어 사용함.

class own_S2S(tf.keras.Model):
    def __init__(self, encoder_dim: int, decoder_dim: int, embedding_dim: int):
        super(own_S2S, self).__init__()
        self.UseEncoderInTest = True

        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.embedding_dim = embedding_dim

        self.enc_inputs = Input(shape=(None, self.encoder_dim))
        self.enc_embedding = Embedding(self.encoder_dim, self.embedding_dim)
        self.enc_masking = Masking(mask_value=0.0)  # masking <pad>
        self.enc_lstm = LSTM(256, return_state=True)  # return_state > return last hidden/cell state one more

        self.dec_inputs = Input(shape=(None, self.decoder_dim))
        self.dec_embedding = Embedding(self.decoder_dim, self.embedding_dim)
        self.dec_masking = Masking(mask_value=0.0)  # masking <pad>
        self.dec_lstm = LSTM(256, return_state=True, return_sequences=True)  # return_sequences > return all step's hidden_state

        self.dec_softmax = Dense(decoder_dim, activation="softmax")

        self.test_decoder_state_input_h = tf.keras.layers.Input(shape=(self.embedding_size,))
        self.test_decoder_state_input_c = tf.keras.layers.Input(shape=(self.embedding_size,))

    def call(self, x, training=None, *args, **kwargs):
        if training:
            encoder_input, decoder_input = x

            encoder_input = self.enc_inputs(encoder_input)
            encoder_input = self.enc_embedding(encoder_input)
            encoder_input = self.enc_masking(encoder_input)
            encoder_outputs, state_h, state_c = self.enc_lstm(encoder_input)
            encoder_state = [state_h, state_c]

            decoder_input = self.dec_inputs(decoder_input)
            decoder_input = self.dec_embedding(decoder_input)
            decoder_input = self.dec_masking(decoder_input)
            decoder_outputs, state_h, state_c = self.dec_lstm(decoder_input, initial_state=encoder_state)

            return self.dec_softmax(decoder_outputs)
        else:
            if self.UseEncoderInTest:
                encoder_input, decoder_input = x

                encoder_input = self.enc_inputs(x)
                encoder_input = self.enc_embedding(encoder_input)
                encoder_input = self.enc_masking(encoder_input)
                encoder_outputs, state_h, state_c = self.enc_lstm(encoder_input)
                encoder_state = [state_h, state_c]

                decoder_input = self.dec_inputs(decoder_input)
                decoder_input = self.dec_embedding(decoder_input)
                decoder_outputs, decoder_state_h, decoder_state_c = self.dec_lstm(decoder_input, initial_state=encoder_state)
            else:
                decoder_input, input_state_h, input_state_c = x

                decoder_state = [self.test_decoder_state_input_h(input_state_h), self.test_decoder_state_input_c(input_state_c)]

                decoder_input = self.dec_inputs(decoder_input)
                decoder_input = self.dec_embedding(decoder_input)
                decoder_outputs, decoder_state_h, decoder_state_c = self.dec_lstm(decoder_input, initial_state=decoder_state)

            return self.dec_softmax(decoder_outputs), decoder_state_h, decoder_state_c

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data

        y_pred = []
        temp_pred = decoder_output = state_h = state_c = 0
        self.UseEncoderInTest = True
        while temp_pred != 2:  # if result is eos, stop result getting
            if self.UseEncoderInTest:
                # in first step, use Encoder with x, and put the result in decoder with <sos>(1)
                self.UseEncoderInTest = False
                decoder_output, state_h, state_c = self((x, 1), training=False)  # have to onehot encoding, but don't do that
            else:
                # in other step, put in decoder that decoder_output, state_h, state_c.
                decoder_output, state_h, state_c = self((decoder_output, state_h, state_c), training=False)

            # decoder_output's is list of float (softmax)
            temp_pred = decoder_output
            y_pred.append(temp_pred)  # require argmax, but don't do that
        y_pred = tf.convert_to_tensor(y_pred)
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

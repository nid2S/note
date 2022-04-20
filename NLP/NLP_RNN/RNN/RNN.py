import numpy as np

# 의사코드
# hidden_state_t = 0
# for input_t in input_length:
#     output_t = tanh(input_t, hidden_state_t)
#     hidden_state_t = output_t

# timestep    : step's num.           sentence(input sequence)'s length.
# input_dim   : input vectos's dim.   word vec's dim.             neural of input layer
# hidden_size : hidden state's size.  same with output dim.       neural of hidden layer, output layer

def RNN(hidden_size, batch_input,
        Wx=None,  # input's weight
        Wh=None,  # hidden state's weight
        bh=None,  # hidden layer's bias
        ):
    timestep = np.shape(batch_input)[1]
    input_dim = np.shape(batch_input)[2]

    if Wx is None or Wh is None or bh is None:
        if not (Wx is None and Wh is None and bh is None):
            raise ValueError("please input all weight, bias.")
        Wx = np.random.random((hidden_size, input_dim))     # input's weight
        Wh = np.random.random((hidden_size, hidden_size))   # hidden state's weight
        bh = np.random.random((hidden_size,))               # hidden layer's bias

    total_hidden_state = []  # one layer's output
    hidden_state = np.zeros((hidden_size,))  # before step's output

    for i, input_b in enumerate(batch_input):  # batch
        batch_hidden_state = []  # batch's output
        for input_t in input_b:  # step
            output_t = np.tanh(np.dot(Wx, input_t) + np.dot(Wh, hidden_state) + bh)  # tanh(Wx * Xt + Wh * H(t-1) + b)
            batch_hidden_state.append(output_t)  # save every step's hidden state(output)
            hidden_state = output_t
        # hidden output : last step - (batch size, output_dim) | every step - (batch_size, timesteps, input_dim)
        total_hidden_state.append(batch_hidden_state)  # (batch, timestep, tanh(hidden state))

    # # output | Yt = softmax(WyHt + b) | (bs, ts, hs) > (ts, bs, hs)
    # softmax_param = np.dot(Wy, total_hidden_state) + by  # (time step, batch size, hidden state)
    # Y = np.exp(softmax_param) / np.sum(np.exp(softmax_param))

    return total_hidden_state, Wx, Wh, bh

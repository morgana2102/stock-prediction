import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)
    
class RNN_LSTMModel(nn.Module):
    def __init__(self, input_size=1, rnn_hidden=32, lstm_hidden=32):
        super(RNN_LSTMModel, self).__init__()
        self.rnn = nn.RNN(input_size, rnn_hidden, batch_first=True)
        self.lstm = nn.LSTM(rnn_hidden, lstm_hidden, batch_first=True)
        self.fc = nn.Linear(lstm_hidden, 1)

    def forward(self, x):
        r_out, _ = self.rnn(x)
        l_out, _ = self.lstm(r_out)
        out = l_out[:, -1, :]
        return self.fc(out)
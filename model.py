import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, embed_dim, out_dim, num_layer, dropout=0.1):
        super(MLP, self).__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.num_layer = num_layer
        self.dropout = dropout

        self.embedding = nn.Linear(in_dim, embed_dim)
        self.layers = []
        for i in range(self.num_layer):
            self.layers.append(nn.Linear(embed_dim, embed_dim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(self.dropout))
        self.layers = nn.ModuleList(self.layers)
        self.scoring = nn.Linear(self.embed_dim, out_dim)

    def forward(self, feats):
        feats = feats.mean(dim=1)
        out = self.embedding(feats)
        for layer in self.layers:
            out = layer(out)
        score = self.scoring(out)
        return score


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.input_layer(x)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Get output from the last time step
        return out


class TFT(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, temperature=1):
        super(TFT, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # variable selection:
        self.var_weight = nn.Parameter(torch.FloatTensor(input_size))
        nn.init.normal_(self.var_weight)
        self.temperature = temperature
        
        # temporal layer:
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # gate layer:
        self.gate_nn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)

        layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=2, dim_feedforward=hidden_size, batch_first=True)
        self.attn = nn.TransformerEncoder(layer, num_layers=2)
        
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Variable selection
        # var_weight = torch.sigmoid(self.var_weight / self.temperature).expand(x.size())
        # x *= var_weight

        # Temporal modeling
        x = self.input_layer(x)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))

        # gate = self.gate_nn(out)
        # out = x + torch.softmax(gate, dim=-1) * out

        out = self.attn(out)
        
        out = self.fc(out[:, -1, :])  # Get output from the last time step
        return out
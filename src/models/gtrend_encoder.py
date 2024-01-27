import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=52):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TimeDistributed(nn.Module):
    # Takes any module and stacks the time dimension with the batch dimenison of inputs before applying the module
    # Insipired from https://keras.io/api/layers/recurrent_layers/time_distributed/
    # https://discuss.pytorch.org/t/any-pytorch-function-can-work-as-keras-timedistributed/1346/4
    def __init__(self, module, batch_first=True):
        super(TimeDistributed, self).__init__()
        self.module = module # Can be any layer we wish to apply like Linear, Conv etc
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y

class GTrendEmbedder(nn.Module):
    def __init__(self, forecast_horizon, embedding_dim, use_mask, trend_len, num_trends,  gpu_num):
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.input_linear = TimeDistributed(nn.Linear(num_trends, embedding_dim))
        self.pos_embedding = PositionalEncoding(embedding_dim, max_len=trend_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=4, dropout=0.2)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.use_mask = use_mask
        self.gpu_num = gpu_num

    def _generate_encoder_mask(self, size, forecast_horizon):
        mask = torch.zeros((size, size))
        split = math.gcd(size, forecast_horizon)
        for i in range(0, size, split):
            mask[i:i+split, i:i+split] = 1
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).to('cuda:'+str(self.gpu_num))
        return mask
    
    def _generate_square_subsequent_mask(self, size):
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).to('cuda:'+str(self.gpu_num))
        return mask

    def forward(self, gtrends):
        gtrend_emb = self.input_linear(gtrends.permute(0,2,1))
        gtrend_emb = self.pos_embedding(gtrend_emb.permute(1,0,2))
        input_mask = self._generate_encoder_mask(gtrend_emb.shape[0], self.forecast_horizon)
        if self.use_mask == 1:
            gtrend_emb = self.encoder(gtrend_emb, input_mask)
        else:
            gtrend_emb = self.encoder(gtrend_emb)
        return gtrend_emb
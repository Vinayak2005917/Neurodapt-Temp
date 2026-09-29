import torch
import json
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence



# Clause Segmentation Model
#Input projection layer: 1536 -> 384
#Transformer Encoder: d_model=384, nhead=4, num_layers=4, dropout=0.2
#Prediction Neural Network layer: 384 -> 1 (for each token)
class ClauseSegmentationModel(nn.Module):
    def __init__(self,input_dim=1536, d_model=384, nhead=4, num_layers=4, dropout=0.2):
        super().__init__()#pytorch stuff, gotta write everytime

        #1536 -> 384 (since 1536/4 = 384, and 384 is divisible by 4)
        self.input_projection = nn.Linear(input_dim, d_model)

        #define the transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            dropout=dropout,
            batch_first=True
        )

        #num_layers set on top
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # Prediction for *each* token
        self.classifier = nn.Linear(d_model, 1)


    def forward(self, embeddings, padding_mask=None):

        # embeddings: [batch_size, sentence_length, input_dim=1536]
        x = self.input_projection(embeddings)

        #Processing the embeddings through the transformer
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)#<- to ignore the padded tokens

        logits = self.classifier(x).squeeze(-1)

        return logits
import torch
import math

from torch import nn


class LSTMEncoderModel(nn.Module):
    def __init__(self, token_num: int, word_embed_size: int, hidden_size: int, dropout: float = 0.1):
        super(LSTMEncoderModel, self).__init__()
        self.embedding = nn.Embedding(token_num, word_embed_size, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size
        self.gru_encoder = nn.GRU(
            input_size=word_embed_size, hidden_size=hidden_size, bidirectional=True, batch_first=True
        )

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        embedded_inputs = self.dropout(self.embedding(src))
        outputs, enc_h_t = self.gru_encoder(embedded_inputs)
        return enc_h_t[-1]


class DSSMModel(nn.Module):
    def __init__(self, encoder: LSTMEncoderModel):
        super(DSSMModel, self).__init__()
        self.encoder = encoder
        self.softmax = torch.nn.Softmax(dim=-1)
        self.hidden_size = self.encoder.hidden_size

    def forward(self, contexts, replies):
        encoded_contexts = self.encoder(contexts)
        encoded_replies = self.encoder(replies)

        probs = []
        # Todo: 배치 내의 k-1개를 네가티브로 간주
        for batch_idx in range(encoded_contexts.size()[0]):
            dot_product_values = torch.tensor(
                [
                    torch.mm(
                        encoded_contexts[batch_idx, :].view(1, self.hidden_size),
                        encoded_replies[i, :].view(self.hidden_size, 1),
                    ).squeeze()
                    for i in range(encoded_contexts.size()[0])
                ]
            )
            normalized_dot_product_values = self.softmax(dot_product_values)
            probs.append(normalized_dot_product_values[batch_idx])
        return torch.tensor(probs).unsqueeze(1)

import numpy as np
import torch
from pnlp.text import Vocab
from torch import nn


class RNNEncoderModel(nn.Module):
    def __init__(
        self, vocab: Vocab, word_embed_size: int, hidden_size: int, glove_embed_file_path: str, dropout: float = 0.1
    ):
        super(RNNEncoderModel, self).__init__()
        self.token_num = len(vocab)
        self.word_embed_size = word_embed_size

        self.embedding = nn.Embedding(self.token_num, word_embed_size, padding_idx=0)
        self.embedding.weight.data.copy_(torch.from_numpy(self._load_glove_embeddings(vocab, glove_embed_file_path)))
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size
        self.gru_encoder = nn.GRU(
            input_size=word_embed_size, hidden_size=hidden_size, bidirectional=True, batch_first=True
        )

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        embedded_inputs = self.dropout(self.embedding(src))
        _, enc_h_t = self.gru_encoder(embedded_inputs)
        return enc_h_t[-1]

    def _load_glove_embeddings(self, vocab: Vocab, file_path: str = "data/glove.6B.100d.txt") -> np.ndarray:
        embeddings = {}
        with open(file_path) as f:
            for line in f:
                splits = line.strip().split()
                word = splits[0]
                embedding = [float(i) for i in splits[1:]]
                if word in vocab:
                    embeddings[word] = embedding

        weights_matrix = np.random.uniform(-0.25, 0.25, size=(self.token_num, self.word_embed_size))
        for word in embeddings:
            weights_matrix[vocab.convert_token_to_id(word)] = embeddings[word]
        return weights_matrix


class ConveRTModel(nn.Module):
    def __init__(
        self, query_encoder: RNNEncoderModel, context_encoder: RNNEncoderModel, reply_encoder: RNNEncoderModel,
    ):
        super(ConveRTModel, self).__init__()
        self.query_encoder = query_encoder
        self.context_encoder = context_encoder
        self.reply_encoder = reply_encoder

        assert self.query_encoder.hidden_size == self.context_encoder.hidden_size == self.reply_encoder.hidden_size
        self.softmax = torch.nn.Softmax(dim=-1)
        self.hidden_size = self.query_encoder.hidden_size
        self.linear = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, query, context, reply):
        encoded_query = self.query_encoder(query)
        encoded_context = self.context_encoder(context)
        encoded_reply = self.reply_encoder(reply)
        batch_size = encoded_context.size()[0]

        encoded_qc = self.linear(encoded_query.add(encoded_context) / 2)

        interaction_between_q_r_probs = []
        interaction_between_c_r_probs = []
        interaction_between_qc_r_probs = []
        # 배치 내의 k-1개를 네가티브로 간주
        # interaction between query and reply
        for batch_idx in range(batch_size):
            dot_product_values = torch.stack(
                [
                    torch.mm(
                        encoded_query[batch_idx, :].view(1, self.hidden_size),
                        encoded_reply[i, :].view(self.hidden_size, 1),
                    ).squeeze()
                    for i in range(batch_size)
                ],
            )
            normalized_dot_product_values = self.softmax(dot_product_values)
            interaction_between_q_r_probs.append(normalized_dot_product_values)

        # interaction between context and reply
        for batch_idx in range(batch_size):
            dot_product_values = torch.stack(
                [
                    torch.mm(
                        encoded_context[batch_idx, :].view(1, self.hidden_size),
                        encoded_reply[i, :].view(self.hidden_size, 1),
                    ).squeeze()
                    for i in range(batch_size)
                ],
            )
            normalized_dot_product_values = self.softmax(dot_product_values)
            interaction_between_c_r_probs.append(normalized_dot_product_values)

        # interaction between q_c and reply
        for batch_idx in range(batch_size):
            dot_product_values = torch.stack(
                [
                    torch.mm(
                        encoded_qc[batch_idx, :].view(1, self.hidden_size),
                        encoded_reply[i, :].view(self.hidden_size, 1),
                    ).squeeze()
                    for i in range(batch_size)
                ],
            )
            normalized_dot_product_values = self.softmax(dot_product_values)
            interaction_between_qc_r_probs.append(normalized_dot_product_values)

        return (
            torch.stack(interaction_between_q_r_probs),
            torch.stack(interaction_between_c_r_probs),
            torch.stack(interaction_between_qc_r_probs),
        )

    def validate_forward(self, query: torch.Tensor, context: torch.Tensor, candidates: torch.Tensor):
        encoded_query = self.query_encoder(query)
        encoded_context = self.context_encoder(context)
        encoded_candidates = torch.stack([self.reply_encoder(candidates[:, i, :]) for i in range(candidates.size()[1])])
        batch_size = encoded_context.size()[0]

        encoded_qc = self.linear(encoded_query.add(encoded_context) / 2)

        probs = []
        for batch_idx in range(batch_size):
            dot_product_values = torch.stack(
                [
                    torch.mm(
                        encoded_qc[batch_idx, :].view(1, self.hidden_size),
                        encoded_candidates[i, batch_idx, :].view(self.hidden_size, 1),
                    ).squeeze()
                    for i in range(encoded_candidates.size()[0])
                ]
            )
            normalized_dot_product_values = self.softmax(dot_product_values)
            probs.append(normalized_dot_product_values)
        return torch.stack(probs)

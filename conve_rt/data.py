import csv
from typing import List, Tuple

import torch
from pnlp.pipeline import NLPPipeline
from pnlp.text import Vocab
from torch.utils.data import Dataset


def load_dataset(file_path: str) -> List[List[str]]:
    reader = csv.reader(open(file_path))
    return list(reader)[1:]


class DSSMTrainDataset(Dataset):
    def __init__(
        self, file_path: str, max_len: int, token_vocab: Vocab, pipeline: NLPPipeline,
    ):
        """데이터셋을 읽고, Model의 입력 형태로 변환해주는 Dataset입니다."""
        self.max_len = max_len
        self.token_vocab = token_vocab
        self.pipeline = pipeline
        self.training_instances = self._create_training_instances(file_path)

    def __len__(self) -> int:
        return len(self.training_instances)

    def __getitem__(self, key: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.training_instances[key]

    def _create_training_instances(self, file_path: str) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """데이터셋의 경로를 받아 각 데이터를 Model의 입력 형태로 변환하여 리스트 형태로 반환해주는 함수입니다."""
        instances = []
        for line in load_dataset(file_path):
            if line[2] == "0":
                continue
            # for singleturn
            context = line[0].split("__eou__")[-2]
            tokenized_context = self.pipeline.run(context)
            reply = line[1]
            tokenized_reply = self.pipeline.run(reply)

            truncated_context = tokenized_context[-self.max_len :]
            truncated_reply = tokenized_reply[-self.max_len :]

            featurized_context = self.token_vocab.convert_tokens_to_ids(truncated_context)
            featurized_reply = self.token_vocab.convert_tokens_to_ids(truncated_reply)

            padded_context = featurized_context + [self.token_vocab.convert_token_to_id("<PAD>")] * (
                self.max_len - len(featurized_context)
            )
            padded_reply = featurized_reply + [self.token_vocab.convert_token_to_id("<PAD>")] * (
                self.max_len - len(featurized_reply)
            )
            instances.append(
                (
                    torch.tensor(padded_context, dtype=torch.long),
                    torch.tensor(padded_reply, dtype=torch.long),
                )
            )
        return instances


class DSSMEvalDataset(Dataset):
    def __init__(
        self, file_path: str, max_len: int, token_vocab: Vocab, pipeline: NLPPipeline,
    ):
        """데이터셋을 읽고, Model의 입력 형태로 변환해주는 Dataset입니다."""
        self.max_len = max_len
        self.token_vocab = token_vocab
        self.pipeline = pipeline
        self.eval_instances = self._create_eval_instances(file_path)

    def __len__(self) -> int:
        return len(self.eval_instances)

    def __getitem__(self, key: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.eval_instances[key]

    def _create_eval_instances(self, file_path: str) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """데이터셋의 경로를 받아 각 데이터를 Model의 입력 형태로 변환하여 리스트 형태로 반환해주는 함수입니다."""
        instances = [self._create_eval_single_instance(row) for row in load_dataset(file_path)]
        return instances

    def _create_eval_single_instance(self, line: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context = line[0]
        tokenized_context = self.pipeline.run(context)
        reply = line[1]
        tokenized_reply = self.pipeline.run(reply)
        tokenized_distractors = [self.pipeline.run(distractor) for distractor in line[2:]]

        truncated_context = tokenized_context[-self.max_len :]
        truncated_reply = tokenized_reply[-self.max_len :]
        truncated_distractors = [
            tokenized_distractor[-self.max_len :] for tokenized_distractor in tokenized_distractors
        ]

        featurized_context = self.token_vocab.convert_tokens_to_ids(truncated_context)
        featurized_reply = self.token_vocab.convert_tokens_to_ids(truncated_reply)
        featurized_distractors = [
            self.token_vocab.convert_tokens_to_ids(truncated_distractor)
            for truncated_distractor in truncated_distractors
        ]

        padded_context = featurized_context + [self.token_vocab.convert_token_to_id("<PAD>")] * (
            self.max_len - len(featurized_context)
        )
        padded_reply = featurized_reply + [self.token_vocab.convert_token_to_id("<PAD>")] * (
            self.max_len - len(featurized_reply)
        )
        padded_distractors = [
            featurized_distractor
            + [self.token_vocab.convert_token_to_id("<PAD>")] * (self.max_len - len(featurized_distractor))
            for featurized_distractor in featurized_distractors
        ]
        return (
            torch.tensor(padded_context, dtype=torch.long),
            torch.tensor(padded_reply, dtype=torch.long),
            torch.tensor(padded_distractors, dtype=torch.float),
        )

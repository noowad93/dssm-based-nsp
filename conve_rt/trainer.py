from logging import Logger
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.optim.adam import Adam
from torch.utils.data import DataLoader

from conve_rt.config import TrainConfig
from conve_rt.model import DSSMModel


class Trainer:
    def __init__(
        self,
        config: TrainConfig,
        model: DSSMModel,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        logger: Logger,
    ):
        """
        모델을 학습시키기 위한 로직을 관리하는 클래스입니다.
        """
        self.config = config
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.logger = logger

        self.device, self.list_ids = self._prepare_device(config.n_gpu, logger)
        self.model.to(self.device)
        # if len(self.list_ids) > 1:
        #     self.model = nn.DataParallel(self.model, device_ids=self.list_ids)
        self.optimizer = Adam(model.parameters(), lr=config.learning_rate)
        self.criterion = (
            nn.CrossEntropyLoss() if self.config.label_smoothing_value == 0.0 else nn.KLDivLoss(reduction="batchmean")
        )
        self.criterion.to(self.device)

        self.steps_per_epoch = len(train_dataloader)
        self.total_steps = self.steps_per_epoch * config.epoch

    def train(self):
        self.logger.info("========= Start of train config ========")
        self.logger.info(f"device                : {self.device}")
        self.logger.info(f"dataset length/ train : {len(self.train_dataloader.dataset)}")
        self.logger.info(f"dataset length/ test  : {len(self.eval_dataloader.dataset)}")
        self.logger.info(f"max sequence length   : {self.config.max_seq_len}")
        self.logger.info(f"train batch size      : {self.config.train_batch_size}")
        self.logger.info(f"learning rate         : {self.config.learning_rate}")
        self.logger.info(f"dropout prob          : {self.config.dropout_prob}")
        self.logger.info(f"total epoch           : {self.config.epoch}")
        self.logger.info(f"steps per epoch       : {self.steps_per_epoch}")
        self.logger.info(f"total steps           : {self.total_steps}")
        self.logger.info("========= End of train config ========")
        global_step = 0
        for epoch in range(1, self.config.epoch + 1):
            loss_sum = 0.0
            for data in self.train_dataloader:
                self.model.train()
                batch_size = data[0].size()[0]
                global_step += 1
                data = tuple(datum.to(self.device) for datum in data)
                self.optimizer.zero_grad()

                outputs = self.model.forward(data[0], data[1])
                if self.config.label_smoothing_value == 0.0:
                    target_labels = torch.arange(batch_size).to(self.device)
                else:
                    outputs = torch.log(outputs)
                    target_labels = get_smoothing_labels(batch_size, self.config.label_smoothing_value).to(self.device)

                loss = self.criterion(outputs, target_labels)

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()

                loss_sum += loss.item()
                if global_step % self.config.train_log_interval == 0:
                    mean_loss = loss_sum / self.config.train_log_interval
                    self.logger.info(f"Epoch {epoch} Step {global_step} Loss {mean_loss:.4f}")
                    loss_sum = 0.0
                if global_step % self.config.val_log_interval == 0:
                    self._validate(global_step)
                if global_step % self.config.save_interval == 0:
                    self._save_model(self.model, global_step)

    def _validate(self, global_step):
        self.model.eval()
        correct_top_k = [0] * 5
        total_instance_num = 0
        with torch.no_grad():
            for data in self.eval_dataloader:
                batch_size = data[0].size()[0]
                total_instance_num += batch_size
                data = tuple(datum.to(self.device) for datum in data)

                outputs = self.model.validate_forward(data[0], data[1])

                _, arg_top_ks = torch.topk(outputs, k=5)
                correct_tensor = arg_top_ks.transpose(0, 1).eq(0)

                for k in range(5):
                    correct_top_k[k] += int(torch.sum(correct_tensor[: k + 1]))

            acc_at_1 = float(correct_top_k[0]) / total_instance_num
            acc_at_5 = float(correct_top_k[4]) / total_instance_num
            self.logger.info(f"[Validation] Acc@1 {acc_at_1:.4f} Acc@5 {acc_at_5:.4f}")

    def _prepare_device(self, n_gpu_use: int, logger: Logger) -> Tuple[torch.device, List[int]]:
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            logger.warn("Warning: There's no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            logger.warn(
                "Warning: The number of GPU's configured to use is {}, but only {} are available "
                "on this machine.".format(n_gpu_use, n_gpu)
            )
            n_gpu_use = n_gpu
        device = torch.device("cuda:0" if n_gpu_use > 0 else "cpu")
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_model(self, model: nn.Module, step: int):
        """모델을 지정된 경로에 저장하는 함수입니다."""
        if isinstance(model, nn.DataParallel):
            torch.save(model.module.state_dict(), f"{self.config.save_model_file_prefix}_step_{step}.pth")
        else:
            torch.save(model.state_dict(), f"{self.config.save_model_file_prefix}_step_{step}.pth")


def get_smoothing_labels(batch_size: int, label_smoothing_value: float) -> torch.Tensor:
    labels = torch.zeros(batch_size, batch_size)
    for i in range(batch_size):
        labels[i].fill_(label_smoothing_value / (batch_size - 1))
        labels[i][i] = 1.0 - label_smoothing_value
    return labels

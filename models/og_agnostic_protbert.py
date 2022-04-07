from itertools import chain

import numpy as np
import torch
from pytorch_lightning import LightningModule
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup
from torch import optim
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score, average_precision_score


class AdvProtBertModule(LightningModule):
    def __init__(
        self,
        protbert_name: str = "Rostlab/prot_bert_bfd",
        max_epochs: int = 100,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 2,
        adv_weight: float = 0.5,
        num_cogs: int = 5000,
    ):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(
            protbert_name, do_lower_case=False
        )
        self.model = BertModel.from_pretrained(protbert_name)
        self.clf_head = nn.Sequential(nn.Linear(self.model.config.hidden_size, 1))
        self.og_head = nn.Sequential(nn.Linear(self.model.config.hidden_size, num_cogs))
        self.save_hyperparameters()

    def forward(
        self, input_ids: torch.LongTensor, attention_mask: torch.Tensor, **kwargs
    ):
        embs = self.model(
            input_ids=input_ids, attention_mask=attention_mask
        ).pooler_output
        return self.clf_head(embs), self.og_head(embs)

    def training_step(self, batch, batch_idx, optimizer_idx):
        embs = self.model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        ).pooler_output

        # train classifier
        if optimizer_idx == 0:
            logits = self.clf_head(embs)
            clf_loss = F.binary_cross_entropy_with_logits(
                logits.view(-1), batch["labels"]
            )
            adv_og_loss = -F.binary_cross_entropy_with_logits(
                self.og_head(embs), batch["og_labels"]
            )
            self.log("train_clf_loss", clf_loss)
            self.log("train_adv_og_loss", -adv_og_loss)
            loss = (
                clf_loss * (1 - self.hparams.adv_weight)
                + adv_og_loss * self.hparams.adv_weight
            )
            self.log("train_regularized_clf_loss", loss)
            return {"loss": loss, "logits": logits.detach(), "labels": batch["labels"]}

        # train og classifier
        if optimizer_idx == 1:
            logits = self.og_head(embs)
            loss = F.binary_cross_entropy_with_logits(logits, batch["og_labels"])
            return {
                "loss": loss,
                "og_logits": logits.detach(),
                "og_labels": batch["og_labels"],
            }

    def validation_step(self, batch, batch_idx):
        embs = self.model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        ).pooler_output
        logits = self.clf_head(embs)
        og_logits = self.og_head(embs)

        clf_loss = F.binary_cross_entropy_with_logits(logits.view(-1), batch["labels"])
        adv_og_loss = -F.binary_cross_entropy_with_logits(og_logits, batch["og_labels"])
        loss = (
            clf_loss * (1 - self.hparams.adv_weight)
            + adv_og_loss * self.hparams.adv_weight
        )

        self.log("val_clf_loss", clf_loss)
        self.log("val_adv_og_loss", -adv_og_loss)
        self.log("val_regularized_clf_loss", loss)

        return {
            "logits": logits.detach(),
            "labels": batch["labels"],
            "og_logits": og_logits.detach(),
            "og_labels": batch["og_labels"],
        }

    def test_step(self, batch, batch_idx):
        embs = self.model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        ).pooler_output
        logits = self.clf_head(embs)
        og_logits = self.og_head(embs)

        clf_loss = F.binary_cross_entropy_with_logits(logits.view(-1), batch["labels"])
        adv_og_loss = -F.binary_cross_entropy_with_logits(og_logits, batch["og_labels"])
        loss = (
            clf_loss * (1 - self.hparams.adv_weight)
            + adv_og_loss * self.hparams.adv_weight
        )

        self.log("test_clf_loss", clf_loss)
        self.log("test_adv_og_loss", -adv_og_loss)
        self.log("test_regularized_clf_loss", loss)

        return {
            "logits": logits.detach(),
            "labels": batch["labels"],
            "og_logits": og_logits.detach(),
            "og_labels": batch["og_labels"],
        }

    def log_epoch_clf_metrics(self, logits: np.ndarray, labels: np.ndarray, stage: str):
        if len(np.unique(labels)) != 1:

            self.log(
                f"{stage}_roc_auc",
                torch.tensor(roc_auc_score(labels, logits), dtype=torch.float32),
            )
            self.log(
                f"{stage}_ap",
                torch.tensor(
                    average_precision_score(labels, logits), dtype=torch.float32
                ),
            )

    def log_epoch_og_metrics(self, logits: np.ndarray, labels: np.ndarray, stage: str):
        if len(np.unique(labels)) != 1:

            self.log(
                f"{stage}_og_roc_auc",
                torch.tensor(
                    roc_auc_score(
                        labels, logits, multi_class="ovr", average="weighted"
                    ),
                    dtype=torch.float32,
                ),
            )

    def training_epoch_end(self, outputs) -> None:
        outputs = self.all_gather(outputs)
        logits = (
            torch.cat([output["logits"] for output in outputs if "logits" in output])
            .cpu()
            .numpy()
            .reshape(-1)
        )
        labels = (
            torch.cat([output["labels"] for output in outputs if "logits" in output])
            .cpu()
            .numpy()
        )
        self.log_epoch_clf_metrics(logits, labels, "train")

        og_logits = (
            torch.cat(
                [output["og_logits"] for output in outputs if "og_logits" in output]
            )
            .cpu()
            .numpy()
            .reshape(-1)
        )
        og_labels = (
            torch.cat(
                [output["og_labels"] for output in outputs if "og_labels" in output]
            )
            .cpu()
            .numpy()
        )
        self.log_epoch_og_metrics(og_logits, og_labels, "train")

    def validation_epoch_end(self, outputs) -> None:
        outputs = self.all_gather(outputs)
        logits = (
            torch.cat([output["logits"] for output in outputs])
            .cpu()
            .numpy()
            .reshape(-1)
        )
        labels = torch.cat([output["labels"] for output in outputs]).cpu().numpy()
        self.log_epoch_clf_metrics(logits, labels, "val")

        og_logits = (
            torch.cat([output["og_logits"] for output in outputs])
            .cpu()
            .numpy()
            .reshape(-1)
        )
        og_labels = torch.cat([output["og_labels"] for output in outputs]).cpu().numpy()
        self.log_epoch_og_metrics(og_logits, og_labels, "val")

    def test_epoch_end(self, outputs) -> None:
        outputs = self.all_gather(outputs)
        logits = (
            torch.cat([output["logits"] for output in outputs])
            .cpu()
            .numpy()
            .reshape(-1)
        )
        labels = torch.cat([output["labels"] for output in outputs]).cpu().numpy()
        self.log_epoch_clf_metrics(logits, labels, "test")

        og_logits = (
            torch.cat([output["og_logits"] for output in outputs])
            .cpu()
            .numpy()
            .reshape(-1)
        )
        og_labels = torch.cat([output["og_labels"] for output in outputs]).cpu().numpy()
        self.log_epoch_og_metrics(og_logits, og_labels, "test")

    def freeze_first_layers(self, num_layers: int = 24):
        for n, p in self.model.named_parameters():
            if "pooler" not in n:
                if "encoder.layer." in n:
                    i = int(n.split(".")[2])
                    if i < num_layers:
                        p.requires_grad = False
                    else:
                        p.requires_grad = True
                else:
                    p.requires_grad = False
            else:
                p.requires_grad = True

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_clf_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in chain(
                        self.model.named_parameters(), self.clf_head.named_parameters()
                    )
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in chain(
                        self.model.named_parameters(), self.clf_head.named_parameters()
                    )
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer_clf = optim.AdamW(
            optimizer_clf_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
        )
        scheduler_clf = get_linear_schedule_with_warmup(
            optimizer_clf,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.hparams.max_epochs,
        )

        optimizer_og_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.og_head.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.og_head.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer_og = optim.AdamW(
            optimizer_og_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
        )
        scheduler_og = get_linear_schedule_with_warmup(
            optimizer_clf,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.hparams.max_epochs,
        )
        return [optimizer_clf, optimizer_og], [scheduler_clf, scheduler_og]

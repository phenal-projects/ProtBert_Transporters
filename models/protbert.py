import numpy as np
import torch
from pytorch_lightning import LightningModule
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup
from torch import optim
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score, average_precision_score


class ProtBertModule(LightningModule):
    def __init__(
        self,
        protbert_name: str = "Rostlab/prot_bert_bfd",
        max_epochs: int = 100,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 2,
    ):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(
            protbert_name, do_lower_case=False
        )
        self.model = BertModel.from_pretrained(protbert_name)
        self.clf_head = nn.Sequential(nn.Linear(self.model.config.hidden_size, 1))
        self.save_hyperparameters()

    def forward(
        self, input_ids: torch.LongTensor, attention_mask: torch.Tensor, **kwargs
    ):
        return self.clf_head(
            self.model(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        )

    def training_step(self, batch, batch_idx):
        logits = self(**batch)

        loss = F.binary_cross_entropy_with_logits(logits.view(-1), batch["labels"])

        self.log("train_loss", loss)
        return {"loss": loss, "logits": logits, "labels": batch["labels"]}

    def validation_step(self, batch, batch_idx):
        logits = self(**batch)

        loss = F.binary_cross_entropy_with_logits(logits.view(-1), batch["labels"])

        self.log("val_loss", loss)
        return {"logits": logits, "labels": batch["labels"]}

    def test_step(self, batch, batch_idx):
        logits = self(**batch)

        loss = F.binary_cross_entropy_with_logits(logits.view(-1), batch["labels"])

        self.log("test_loss", loss)
        return {"logits": logits, "labels": batch["labels"]}

    def log_epoch_metrics(self, logits: np.ndarray, labels: np.ndarray, stage: str):
        if len(np.unique(labels)) != 1:

            self.log(
                f"{stage}_roc_auc",
                roc_auc_score(labels, logits),
            )
            self.log(
                f"{stage}_ap",
                average_precision_score(labels, logits),
            )

    def training_epoch_end(self, outputs) -> None:
        outputs = self.all_gather(outputs)
        logits = torch.cat([output["logits"] for output in outputs]).cpu().numpy()
        labels = torch.cat([output["labels"] for output in outputs]).cpu().numpy()

        self.log_epoch_metrics(logits, labels, "train")

    def validation_epoch_end(self, outputs) -> None:
        outputs = self.all_gather(outputs)
        logits = torch.cat([output["logits"] for output in outputs]).cpu().numpy()
        labels = torch.cat([output["labels"] for output in outputs]).cpu().numpy()

        self.log_epoch_metrics(logits, labels, "val")

    def test_epoch_end(self, outputs) -> None:
        outputs = self.all_gather(outputs)
        logits = torch.cat([output["logits"] for output in outputs]).cpu().numpy()
        labels = torch.cat([output["labels"] for output in outputs]).cpu().numpy()

        self.log_epoch_metrics(logits, labels, "test")

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
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.hparams.max_epochs,
        )
        return [optimizer], [scheduler]

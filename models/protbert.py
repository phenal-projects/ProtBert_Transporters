import numpy as np
import torch
from pytorch_lightning import LightningModule
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup
from torch import optim
from torch import nn
from torchmetrics.classification import MulticlassAUROC, MulticlassAveragePrecision
from sklearn.metrics import roc_auc_score, average_precision_score


class ProtBertModule(LightningModule):
    def __init__(
        self,
        protbert_name: str = "Rostlab/prot_bert_bfd",
        max_epochs: int = 100,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        adam_epsilon: float = 1e-8,
        num_classes: int = 2,
        warmup_steps: int = 2,
    ):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(
            protbert_name, do_lower_case=False
        )
        self.model = BertModel.from_pretrained(protbert_name)
        if num_classes == 2:
            self.clf_head = nn.Sequential(nn.Linear(self.model.config.hidden_size, 1))
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.clf_head = nn.Sequential(
                nn.Linear(self.model.config.hidden_size, num_classes)
            )
            self.loss_fn = nn.CrossEntropyLoss()
        self.save_hyperparameters()

        self.roc_auc_fn = MulticlassAUROC(num_classes=num_classes, average=None)
        self.ap_fn = MulticlassAveragePrecision(num_classes=num_classes, average=None)

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(
        self, input_ids: torch.LongTensor, attention_mask: torch.Tensor, **kwargs
    ):
        return self.clf_head(
            self.model(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        )

    def training_step(self, batch, batch_idx):
        logits = self(**batch)

        loss = self.loss_fn(logits, batch["labels"])

        self.log("train_loss", loss)
        self.training_step_outputs.append(
            (logits.detach().cpu(), batch["labels"].cpu())
        )
        return {"loss": loss, "logits": logits.detach(), "labels": batch["labels"]}

    def validation_step(self, batch, batch_idx):
        logits = self(**batch)

        loss = self.loss_fn(logits, batch["labels"])

        self.log("val_loss", loss)
        self.validation_step_outputs.append(
            (logits.detach().cpu(), batch["labels"].cpu())
        )
        return {"logits": logits, "labels": batch["labels"]}

    def test_step(self, batch, batch_idx):
        logits = self(**batch)

        loss = self.loss_fn(logits, batch["labels"])
        self.test_step_outputs.append((logits.detach().cpu(), batch["labels"].cpu()))
        self.log("test_loss", loss)
        return {"logits": logits, "labels": batch["labels"]}

    def on_train_epoch_end(self):
        logits = torch.cat([x[0] for x in self.training_step_outputs]).float()
        labels = torch.cat([x[1] for x in self.training_step_outputs])
        self.log_epoch_metrics(logits, labels, "train")
        self.training_step_outputs.clear()

    def on_validation_epoch_end(self):
        logits = torch.cat([x[0] for x in self.validation_step_outputs]).float()
        labels = torch.cat([x[1] for x in self.validation_step_outputs])
        self.log_epoch_metrics(logits, labels, "val")
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        logits = torch.cat([x[0] for x in self.test_step_outputs]).float()
        labels = torch.cat([x[1] for x in self.test_step_outputs])
        self.log_epoch_metrics(logits, labels, "test")
        self.test_step_outputs.clear()

    def log_epoch_metrics(self, logits, labels, stage: str):
        roc_aucs = self.roc_auc_fn(logits, labels)
        for i, metric in enumerate(roc_aucs):
            self.log(
                f"{stage}_roc_auc_{i}",
                metric,
            )
        self.log(
            f"{stage}_roc_auc",
            roc_aucs.mean(),
        )
        for i, metric in enumerate(self.ap_fn(logits, labels)):
            self.log(
                f"{stage}_ap_{i}",
                metric,
            )

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

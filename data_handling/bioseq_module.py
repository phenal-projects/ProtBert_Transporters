import os
from typing import Union, Optional, Callable, Collection

import numpy as np
from bioseq_dataset import SequenceData, SequenceDataset
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from pytorch_lightning import LightningDataModule

from transformers import DataCollatorWithPadding, PreTrainedTokenizer


class SequenceLMDBDataset(Dataset):
    def __init__(
        self,
        db_path: Union[str, os.PathLike],
        transform: Optional[Callable] = None,
        keys: Optional[Collection[str]] = None,
    ):
        self.db = SequenceDataset(
            db_path,
            read_only=True,
        )
        if keys is not None:
            self.keys = sorted(keys)
        else:
            self.keys = None
        self.transform = transform

    def init_db(self):
        self.db.init_db()
        if self.keys is None:
            self.keys = sorted(self.db.get_keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, item):
        x = self.db.get_sequence(self.keys[item])
        if self.transform is not None:
            x = self.transform(x)
        return x


class SequencesDataModule(LightningDataModule):
    def __init__(
        self,
        db_path: Union[str, os.PathLike],
        tokenizer: PreTrainedTokenizer,
        val_classes: Collection[str],
        test_classes: Collection[str],
        training_batches_per_epoch: int = 5000,
        batch_size: int = 10,
        random_seed: int = 42,
    ):
        super().__init__()
        self.db_path = db_path
        self.db = SequenceDataset(db_path)
        self.tokenizer = tokenizer

        self.training_batches_per_epoch = training_batches_per_epoch
        self.batch_size = batch_size
        self.random_seed = random_seed

        self.test_classes = set(test_classes)
        self.val_classes = set(val_classes)

        self.datasets = None
        self.cog_dict = None
        self.train_sampling_weights = None

    def tokenize(self, x: SequenceData):
        dct = self.tokenizer(" ".join(x.seq.upper()), truncation=True, max_length=1024)
        dct["labels"] = float(x.label)
        if self.cog_dict is not None:
            dct["og_labels"] = [0.0] * len(self.cog_dict)
            for k in x.seq_classes:
                if k in dct["og_labels"]:
                    dct["og_labels"][self.cog_dict[k]] = 1.0
        return dct

    def setup(self, stage: Optional[str] = None) -> None:
        self.db.init_db()
        keys = sorted(self.db.get_keys())

        train, val, test = [], [], []
        for key in keys:
            if len(self.db[key].seq_classes & self.test_classes) > 0:
                test.append(key)
            elif len(self.db[key].seq_classes & self.val_classes) > 0:
                val.append(key)
            else:
                train.append(key)

        # fit cogs idx
        all_cogs = set()
        for key in keys:
            all_cogs.update(self.db[key].seq_classes)
        self.cog_dict = {k: i for i, k in enumerate(sorted(all_cogs))}

        self.datasets = dict()
        if stage == "fit" or stage is None:
            self.datasets["train"] = SequenceLMDBDataset(
                self.db_path, self.tokenize, keys=train
            )
            self.datasets["val"] = SequenceLMDBDataset(
                self.db_path, self.tokenize, keys=val
            )
            self.datasets["train"].init_db()
            self.datasets["val"].init_db()

            train_labels = np.array(
                [self.db[key].label for key in self.datasets["train"].keys]
            )
            weights = {0: train_labels.mean(), 1: 1 - train_labels.mean()}
            self.train_sampling_weights = [weights[label] for label in train_labels]
        if stage == "test" or stage is None:
            self.datasets["test"] = SequenceLMDBDataset(
                self.db_path, self.tokenize, keys=test
            )
            self.datasets["test"].init_db()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.datasets["train"],
            num_workers=4,
            sampler=WeightedRandomSampler(
                weights=self.train_sampling_weights,
                num_samples=self.batch_size*self.training_batches_per_epoch,
                replacement=False,
            ),
            batch_size=self.batch_size,
            collate_fn=DataCollatorWithPadding(self.tokenizer, padding=True),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.datasets["val"],
            num_workers=4,
            batch_size=self.batch_size,
            collate_fn=DataCollatorWithPadding(self.tokenizer, padding=True),
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.datasets["test"],
            num_workers=4,
            batch_size=self.batch_size,
            collate_fn=DataCollatorWithPadding(self.tokenizer, padding=True),
        )

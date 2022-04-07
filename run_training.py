import os.path

import hydra
import pytorch_lightning as pl
from bioseq_dataset import SequenceDataset
from omegaconf import DictConfig
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from data_handling.bioseq_module import SequencesDataModule
from models.og_agnostic_protbert import AdvProtBertModule
from models.protbert import ProtBertModule


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg["random_seed"])

    # model
    if cfg["training"]["adv_weight"] > 0:
        # get OGs number
        sd = SequenceDataset(
            cfg["data"]["db_path"],
            read_only=True,
        )
        sd.init_db()
        all_cogs = set()
        for key in sd.get_keys():
            all_cogs.update(sd[key].seq_classes)
        del sd

        model = AdvProtBertModule(
            protbert_name=cfg["model"]["name"],
            max_epochs=cfg["training"]["max_epochs"],
            learning_rate=cfg["training"]["learning_rate"],
            weight_decay=cfg["training"]["weight_decay"],
            adam_epsilon=cfg["training"]["adam_epsilon"],
            warmup_steps=cfg["training"]["warmup_steps"],
            adv_weight=cfg["training"]["adv_weight"],
            num_cogs=len(all_cogs)
        )
    else:
        model = ProtBertModule(
            protbert_name=cfg["model"]["name"],
            max_epochs=cfg["training"]["max_epochs"],
            learning_rate=cfg["training"]["learning_rate"],
            weight_decay=cfg["training"]["weight_decay"],
            adam_epsilon=cfg["training"]["adam_epsilon"],
            warmup_steps=cfg["training"]["warmup_steps"],
        )
    model.freeze_first_layers(cfg["model"]["frozen_layers"])
    # data
    data = SequencesDataModule(
        db_path=cfg["data"]["db_path"],
        tokenizer=model.tokenizer,
        val_classes=cfg["data"]["val_cogs"],
        test_classes=cfg["data"]["test_cogs"],
        batch_size=cfg["data"]["batch_size"],
        training_batches_per_epoch=cfg["training"]["training_batches_per_epoch"],
        random_seed=cfg["random_seed"],
    )

    logger = WandbLogger(
        project=cfg["tracking"]["experiment_name"],
        log_model=True,
    )
    logger.log_hyperparams(
        {
            "folder": os.path.abspath(cfg["tracking"]["checkpoints_folder"]),
        }
    )

    if cfg["training"]["amp_backend"] == "apex":
        trainer = pl.Trainer(
            default_root_dir=cfg["tracking"]["checkpoints_folder"],
            gpus=cfg["training"]["gpus"],
            precision=cfg["training"]["precision"],
            amp_backend=cfg["training"]["amp_backend"],
            amp_level=cfg["training"]["amp_level"],
            max_epochs=cfg["training"]["max_epochs"],
            callbacks=[
                ModelCheckpoint(
                    save_top_k=3,
                    save_weights_only=True,
                    mode="max",
                    monitor="val_roc_auc",
                ),
                LearningRateMonitor("epoch"),
            ],
            limit_train_batches=cfg["training"]["training_batches_per_epoch"],
            limit_val_batches=cfg["training"]["val_batches_per_epoch"],
            logger=logger,
            fast_dev_run=cfg["training"]["fast_dev"],
        )
    else:
        trainer = pl.Trainer(
            default_root_dir=cfg["tracking"]["checkpoints_folder"],
            gpus=cfg["training"]["gpus"],
            precision=cfg["training"]["precision"],
            amp_backend=cfg["training"]["amp_backend"],
            max_epochs=cfg["training"]["max_epochs"],
            callbacks=[
                ModelCheckpoint(
                    save_top_k=3,
                    save_weights_only=True,
                    mode="max",
                    monitor="val_roc_auc",
                ),
                LearningRateMonitor("epoch"),
            ],
            limit_train_batches=cfg["training"]["training_batches_per_epoch"],
            limit_val_batches=cfg["training"]["val_batches_per_epoch"],
            logger=logger,
            fast_dev_run=cfg["training"]["fast_dev"],
        )
    trainer.fit(model, data)
    trainer.test(model, data)


if __name__ == "__main__":
    main()

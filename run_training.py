import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from data_handling.bioseq_module import SequencesDataModule
from models.protbert import ProtBertModule


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg["random_seed"])

    # model
    model = ProtBertModule(
        protbert_name="Rostlab/prot_bert_bfd",
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
        batch_size=cfg["data"]["batch_size"],
        random_seed=cfg["random_seed"],
    )

    logger = WandbLogger(
        project=cfg["tracking"]["experiment_name"],
        log_model=True,
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
            limit_train_batches=5000,
            limit_val_batches=1000,
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
            limit_train_batches=5000,
            limit_val_batches=1000,
            logger=logger,
            fast_dev_run=cfg["training"]["fast_dev"],
        )
    trainer.fit(model, data)
    trainer.test(model, data)


if __name__ == "__main__":
    main()

__all__ = ['UpliftDataModule']


from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class UpliftDataModule(LightningDataModule):
    """LightningDataModule wrapping train / validation / predict datasets.

    Args:
        train_dataset (Dataset): Training data.
        valid_dataset (Dataset): Optional validation data.
        predict_dataset (Dataset): Dataset used at predict time. Falls back to
            train_dataset when None (supports post-fit Trainer.predict calls).
        batch_size (int): Train batch size.
        valid_batch_size (int): Validation / predict batch size.
        shuffle_train (bool): Shuffle training data each epoch.
        **dataloader_kwargs: Forwarded to every DataLoader.
    """

    def __init__(
        self,
        train_dataset: Dataset | None = None,
        valid_dataset: Dataset | None = None,
        predict_dataset: Dataset | None = None,
        batch_size: int = 32,
        valid_batch_size: int | None = 1024,
        shuffle_train: bool = True,
        **dataloader_kwargs,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.predict_dataset = predict_dataset
        self.batch_size = batch_size
        self.valid_batch_size = valid_batch_size or batch_size
        self.shuffle_train = shuffle_train
        self.dataloader_kwargs = dataloader_kwargs

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            **self.dataloader_kwargs,
        )

    def val_dataloader(self) -> DataLoader | list:
        if self.valid_dataset is None:
            return []
        return DataLoader(
            self.valid_dataset,
            batch_size=self.valid_batch_size,
            shuffle=False,
            **self.dataloader_kwargs,
        )

    def predict_dataloader(self) -> DataLoader:
        dataset = self.predict_dataset if self.predict_dataset is not None else self.train_dataset
        return DataLoader(
            dataset,
            batch_size=self.valid_batch_size,
            shuffle=False,
            **self.dataloader_kwargs,
        )

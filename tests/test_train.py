import os
import shutil

import pytest

from molskill.data.dataloaders import get_dataloader
from molskill.data.featurizers import AVAILABLE_FEATURIZERS, get_featurizer
from molskill.models.utils import get_new_model_and_trainer
from molskill.paths import MODEL_PATH


@pytest.mark.parametrize("featurizer_name", list(AVAILABLE_FEATURIZERS.keys()))
def test_loss_goes_down(pair_training_data, n_epochs, num_workers, featurizer_name):
    molrpr, target = pair_training_data

    test_model_dir = os.path.join(MODEL_PATH, "test")
    if os.path.exists(test_model_dir):
        shutil.rmtree(test_model_dir)

    featurizer = get_featurizer(featurizer_name)
    dataloader = get_dataloader(
        molrpr=molrpr,
        target=target,
        shuffle=False,
        featurizer=featurizer,
        num_workers=num_workers,
    )
    model, trainer = get_new_model_and_trainer(
        save_dir=test_model_dir,
        n_epochs=n_epochs,
        input_size=featurizer.dim(),
    )

    initial_loss = trainer.validate(model, dataloaders=dataloader)[0]["val/loss"]

    trainer.fit(model, train_dataloaders=dataloader)
    loss = trainer.callback_metrics["train/loss"].item()
    shutil.rmtree(test_model_dir)
    assert loss < initial_loss

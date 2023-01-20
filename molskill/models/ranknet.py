from functools import partial
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.optim import Adam

_NOT_RECOGNISED_INPUT_TYPE = ValueError(
    "Not recognised input format, should be either tensor or tuple of tensors"
)

BatchPairNoTarget = Tuple[torch.Tensor, torch.Tensor]
BatchPairTarget = Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]


def compute_metrics(
    logit: torch.Tensor,
    target: torch.Tensor,
    loss: Optional[torch.Tensor] = None,
    prob_thresh: float = 0.5,
    target_prob: bool = False,
) -> Dict[str, torch.Tensor]:
    """Computes binary classification metrics

    Args:
        logit (torch.Tensor): output logits from model
        target (torch.Tensor): target labels
        loss (Optional[torch.Tensor], optional): loss val. Defaults to None.
        prob_thresh (float): Probability label threshold. Defaults to 0.5
        target_prob (bool): Whether the target is a probability or a binary label.

    Returns:
        Dict[str, torch.Tensor]: metric dictionary, only return not None values
    """
    metrics = {"loss": loss, "auroc": None, "acc": None}

    with torch.no_grad():
        if target_prob:
            target = target > prob_thresh
        target = target.cpu().numpy()[:, 0]
        prob = torch.sigmoid(logit).cpu().numpy()[:, 0]
        if len(np.unique(target)) > 1:
            metrics["auroc"] = roc_auc_score(target, prob)

        pred_label = prob > prob_thresh
        metrics["acc"] = (target == pred_label).mean()
    return {
        metric_k: metric_v
        for metric_k, metric_v in metrics.items()
        if metric_v is not None
    }


class RankNet(nn.Module):
    def __init__(
        self,
        input_size: int = 2048,
        hidden_size: int = 256,
        n_layers: int = 3,
        dropout_p: float = 0.0,
    ) -> None:
        """Basic RankNet implementation. Pairs of samples are classified
        according to sigmoid(s_i - s_j) where s_i, s_j are scores learned
        during training.

        Args:
            input_size (int, optional): Descriptor size for each sample. Defaults to 2048.
            hidden_size (int, optional): Number of neurons in hidden layers. Defaults to 256.
            n_layers (int, optional): Number of hidden layers. Defaults to 3.
            dropout_p (float, optional): Dropout probability. Defaults to 0.0.
        """
        super(RankNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.Dropout(dropout_p), nn.ReLU()
        )

        for _ in range(n_layers):
            self.encoder.append(nn.Linear(hidden_size, hidden_size))
            self.encoder.append(nn.Dropout(dropout_p))
            self.encoder.append(nn.ReLU())
        self.encoder.append(nn.Linear(hidden_size, 1))

    def forward(
        self, x_i: torch.Tensor, x_j: torch.Tensor, sigmoid: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        score_i, score_j = self.encoder(x_i), self.encoder(x_j)
        out = score_i - score_j
        if sigmoid:
            out = torch.sigmoid(out)
        return score_i, score_j, out

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """Scores sample `x`

        Args:
            x: input fingerprints // (n_samples, n_feat)
        """
        with torch.inference_mode():
            return self.encoder(x)


class LitRankNet(pl.LightningModule):
    def __init__(
        self,
        net: Optional[RankNet] = None,
        input_size: int = 2048,
        loss_fn: Callable = F.binary_cross_entropy_with_logits,
        regularization_factor: float = 0.0,
        lr: float = 3e-4,
        target_prob: bool = False,
        dropout_p: float = 0.0,
        mc_dropout_samples: int = 1,
        sigmoid: bool = False,
    ) -> None:
        """Main RankNet Lightning module

        Args:
            net (Optional[RankNet], optional): RankNet module. Defaults to None.
            input_size (int, optional): Feature input size for the trained `RankNet` model.
            loss_fn (Callable, optional): loss function to use when training. Defaults to F.binary_cross_entropy_with_logits.
            regularization_factor (float, optional): If > 0.0, whether to apply a regularization factor on the\
                                                     learned score, to ensure zero-symmetry. This is done by\
                                                     adding `normalization_factor *||s||**2` on top of the standard\
                                                     BCE loss.
            lr (float, optional): Learning rate. Defaults to 3e-4.
            target_prob (bool): Whether the target is a probability or a binary label.
            dropout_p (float, optional): Dropout probability. Defaults to 0.0.
            mc_dropout_samples (int, optional): Number of MC dropout samples. Defaults to 1.
            sigmoid (bool, optional): Whether to use a sigmoid function on the output. Defaults to False.
        """
        super().__init__()
        self.net = RankNet(input_size=input_size) if net is None else net
        self.loss_fn = loss_fn
        self.regularization_factor = regularization_factor
        self.lr = lr
        self.target_prob = target_prob
        self.dropout_p = dropout_p
        self.mc_dropout_samples = mc_dropout_samples
        self.sigmoid = sigmoid

    @staticmethod
    def get_reg_loss(
        score_i: torch.Tensor, score_j: torch.Tensor, regularization_factor: float
    ) -> torch.Tensor:
        """Returns regularization loss for the scores ||s||^2 / batch_size
        and scales it by `regularization_factor`
        """
        batch_size = score_i.size(0)
        reg_loss = (
            regularization_factor
            * (torch.norm(score_i) ** 2 + torch.norm(score_j) ** 2)
            / batch_size
        )
        return reg_loss

    def get_scores_logit_target(
        self, batch: BatchPairTarget
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        (x_i, x_j), target = batch
        return self.net(x_i, x_j), target

    def forward(
        self, x_i: torch.Tensor, x_j: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.net(x_i, x_j)

    def score(self, batch) -> torch.Tensor:
        self.net.eval()
        return self.net.score(batch)

    def training_step(
        self, train_batch: BatchPairTarget, batch_idx: int
    ) -> torch.Tensor:
        (score_i, score_j, logit), target = self.get_scores_logit_target(train_batch)
        loss = self.loss_fn(logit, target)
        reg_loss = self.get_reg_loss(score_i, score_j, self.regularization_factor)

        metrics = compute_metrics(logit, target, loss, target_prob=self.target_prob)
        for metric_k, metric_val in metrics.items():
            self.log(f"train/{metric_k}", metric_val)
        self.log("train/regloss", reg_loss.item())
        return loss + reg_loss

    def validation_step(
        self, val_batch: BatchPairTarget, batch_idx: int
    ) -> torch.Tensor:
        (score_i, score_j, logit), target = self.get_scores_logit_target(val_batch)
        loss = self.loss_fn(logit, target)
        reg_loss = self.get_reg_loss(score_i, score_j, self.regularization_factor)

        metrics = compute_metrics(logit, target, loss, target_prob=self.target_prob)
        for metric_k, metric_val in metrics.items():
            self.log(f"val/{metric_k}", metric_val)
        self.log("val/regloss", reg_loss.item())
        return loss + reg_loss

    def test_step(
        self, test_batch: BatchPairTarget, batch_idx: int = 0
    ) -> Dict[str, torch.Tensor]:
        (score_i, score_j, logit), target = self.get_scores_logit_target(test_batch)

        reg_loss = self.get_reg_loss(score_i, score_j, self.regularization_factor)

        metrics = compute_metrics(logit, target, target_prob=self.target_prob)
        for metric_k, metric_val in metrics.items():
            self.log(f"test/{metric_k}", metric_val)
        self.log("test/regloss", reg_loss.item())
        return {metric_k: metric_val for metric_k, metric_val in metrics.items()}

    def predict_step(
        self,
        new_batch: Union[
            torch.Tensor,
            BatchPairNoTarget,
            BatchPairTarget,
        ],
        batch_idx: int = 0,
    ) -> Union[BatchPairNoTarget, torch.Tensor]:
        """Test-time prediction method. When a tuple of Tensors is fed to `new_batch`, predictions as score
        differences between the tensors are returned. When a single Tensor is fed, a single score is returned.
        If MC dropout is set, predictions are returned as a tuple (mean, variance).
        Args:
            new_batch (Tuple[Tensor, Tensor] | Tuple[Tuple[Tensor, Tensor], Tensor] | Tensor): batch
            batch_idx (int): batch index

        Returns:
            (np.ndarray): Mean predictions of the batch.
            (np.ndarray, optional): Uncertainty measured as variance of the predictions.
        """
        if isinstance(new_batch, Sequence):
            if isinstance(new_batch[0], Sequence):
                pred_fun = partial(self.net, sigmoid=self.sigmoid)
            elif isinstance(new_batch[0], torch.Tensor):
                pred_fun = self.net.score
            else:
                raise _NOT_RECOGNISED_INPUT_TYPE
            new_batch = new_batch[0]

        elif isinstance(new_batch, torch.Tensor):
            pred_fun = self.net.score
        else:
            raise _NOT_RECOGNISED_INPUT_TYPE

        return self._predict(new_batch, pred_fun=pred_fun)

    def _predict(
        self,
        new_batch: Union[BatchPairNoTarget, torch.Tensor],
        pred_fun: Callable,
        batch_idx: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        dropout_out: List[torch.Tensor] = []
        if self.mc_dropout_samples > 1:
            enable_dropout(self.net, self.dropout_p)

        with torch.no_grad():
            for _ in range(self.mc_dropout_samples):
                out = (
                    pred_fun(*new_batch)
                    if isinstance(new_batch, tuple)
                    else pred_fun(new_batch)
                )
                dropout_out.append(out)

        dropout_out = torch.cat(dropout_out, dim=1)

        out_mean = dropout_out.mean(dim=1)

        if dropout_out.shape[1] > 1:
            out_var = dropout_out.var(dim=1)

            return (
                out_mean,
                out_var,
            )
        else:
            return out_mean

    def configure_optimizers(self):
        opt = Adam(self.net.parameters(), lr=self.lr)
        return opt


def enable_dropout(model: RankNet, dropout_p: float = 0.0):
    """Function to enable dropout layers during test-time"""
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()
            m.p = dropout_p

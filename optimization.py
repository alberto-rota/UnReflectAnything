import os
import warnings

import dill
import numpy as np
import torch
from torch_sgld import SGLD

from logger import get_logger

logger = get_logger(__name__).set_context("OPTIMIZATION")


warnings.filterwarnings(
    "ignore", message="Your application has authenticated using end user credentials"
)


# Custom implementation of Adam optimizer, extending torch.optim.Adam.
class Adam(torch.optim.Adam):
    def __init__(self, *args, **kwargs):
        """
        Initializes the Adam optimizer with the given arguments and keyword arguments.

        Args:
            *args: Variable length argument list for the base class.
            **kwargs: Arbitrary keyword arguments containing optimizer settings.
        """
        super(Adam, self).__init__(*args, **kwargs)
        self.kwargs = kwargs  # Store kwargs to use in the string representation.

    def __str__(self) -> str:
        """
        Generates a string representation of the Adam optimizer, including its configuration.

        Returns:
            str: A formatted string listing the optimizer's configuration.
        """
        modelstr = f"{self.__class__.__name__}(\n"
        for k in self.kwargs.keys():
            modelstr += f"    {k}: {self.kwargs.get(k, '')} \n"
        modelstr += ")"
        return modelstr


# Custom implementation of SGLD optimizer, extending a base SGLD class.
class SGLD(SGLD):
    def __init__(self, *args, **kwargs):
        """
        Initializes the SGLD optimizer, setting the momentum and temperature based on kwargs.
        Note: Inherits from a base SGLD class, not shown in the provided code.

        Args:
            *args: Variable length argument list for the base class.
            **kwargs: Arbitrary keyword arguments, with 'lr' used to set the temperature.
        """
        super(SGLD, self).__init__(
            momentum=0.9, temperature=kwargs["lr"], *args, **kwargs
        )
        self.kwargs = kwargs
        self.kwargs["temperature"] = kwargs[
            "lr"
        ]  # Adjust temperature to match learning rate.
        self.kwargs["momentum"] = 0.9  # Set momentum to a fixed value.

    def __str__(self) -> str:
        """
        Generates a string representation of the SGLD optimizer, including its configuration.

        Returns:
            str: A formatted string listing the optimizer's configuration.
        """
        modelstr = f"{self.__class__.__name__}(\n"
        for k in self.kwargs.keys():
            modelstr += f"    {k}: {self.kwargs.get(k, '')} \n"
        modelstr += ")"
        return modelstr


# Custom implementation of SGD optimizer, extending torch.optim.SGD.
class SGD(torch.optim.SGD):
    def __init__(self, *args, **kwargs):
        """
        Initializes the SGD optimizer with the given arguments and keyword arguments.

        Args:
            *args: Variable length argument list for the base class.
            **kwargs: Arbitrary keyword arguments containing optimizer settings.
        """
        super(SGD, self).__init__(*args, **kwargs)
        self.kwargs = kwargs  # Store kwargs to use in the string representation.

    def __str__(self) -> str:
        """
        Generates a string representation of the SGD optimizer, including its configuration.

        Returns:
            str: A formatted string listing the optimizer's configuration.
        """
        modelstr = f"{self.__class__.__name__}(\n"
        for k in self.kwargs.keys():
            modelstr += f"    {k}: {self.kwargs.get(k, '')} \n"
        modelstr += ")"
        return modelstr


class RMSprop(torch.optim.RMSprop):
    def __init__(self, *args, **kwargs):
        """
        Initializes the RMSprop optimizer with the given arguments and keyword arguments.

        This custom implementation of RMSprop extends the torch.optim.RMSprop class,
        allowing for additional keyword arguments to be stored and used in the string representation.

        Args:
            *args: Variable length argument list for the base RMSprop class.
            **kwargs: Arbitrary keyword arguments containing optimizer settings.
        """
        super(RMSprop, self).__init__(*args, **kwargs)
        self.kwargs = kwargs  # Store kwargs to be used in the __str__ method.

    def __str__(self) -> str:
        """
        Generates a string representation of the RMSprop optimizer, including its configuration.

        Returns:
            str: A formatted string listing the optimizer's configuration.
        """
        modelstr = f"{self.__class__.__name__}(\n"
        for k in self.kwargs.keys():
            modelstr += f"    {k}: {self.kwargs.get(k, '')} \n"
        modelstr += ")"
        return modelstr


class AdamW(torch.optim.AdamW):
    def __init__(self, *args, **kwargs):
        """
        Initializes the AdamW optimizer with the given arguments and keyword arguments.

        This custom implementation of AdamW extends the torch.optim.AdamW class,
        adding functionality to store and utilize additional keyword arguments in its string representation.

        Args:
            *args: Variable length argument list for the base AdamW class.
            **kwargs: Arbitrary keyword arguments containing optimizer settings.
        """
        super(AdamW, self).__init__(*args, **kwargs)
        self.kwargs = kwargs  # Store kwargs for use in __str__ method.

    def __str__(self) -> str:
        """
        Generates a string representation of the AdamW optimizer, including its configuration.

        Returns:
            str: A formatted string listing the optimizer's configuration.
        """
        modelstr = f"{self.__class__.__name__}(\n"
        for k in self.kwargs.keys():
            modelstr += f"    {k}: {self.kwargs.get(k, '')} \n"
        modelstr += ")"
        return modelstr


class Adagrad(torch.optim.Adagrad):
    def __init__(self, *args, **kwargs):
        """
        Initializes the Adagrad optimizer with the given arguments and keyword arguments.

        This custom implementation of Adagrad extends the torch.optim.Adagrad class,
        incorporating functionality to store additional keyword arguments for use in its string representation.

        Args:
            *args: Variable length argument list for the base Adagrad class.
            **kwargs: Arbitrary keyword arguments containing optimizer settings.
        """
        super(Adagrad, self).__init__(*args, **kwargs)
        self.kwargs = kwargs  # Store kwargs for use in __str__ method.

    def __str__(self) -> str:
        """
        Generates a string representation of the Adagrad optimizer, including its configuration.

        Returns:
            str: A formatted string listing the optimizer's configuration.
        """
        modelstr = f"{self.__class__.__name__}(\n"
        for k in self.kwargs.keys():
            modelstr += f"    {k}: {self.kwargs.get(k, '')} \n"
        modelstr += ")"
        return modelstr


def get_norms(params):
    grad_norm = 0.0
    weight_norm = 0.0
    for p in params:
        if p.requires_grad and p.grad is not None:
            param_norm = p.grad.data.norm(2)
            grad_norm += param_norm.item() ** 2
            param_norm = p.data.norm(2)
            weight_norm += param_norm.item() ** 2
    grad_norm = grad_norm ** (1.0 / 2)
    weight_norm = weight_norm ** (1.0 / 2)
    return grad_norm, weight_norm


import torch
from google.cloud import storage


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self,
        patience=3,
        verbose=False,
        delta=0,
        runname=None,
        checkpointpath="checkpoints",
        bucket_name=None,  # Will use GCS_BUCKET_NAME environment variable if None
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            checkpointpath (str): Path to save temporary checkpoints
            wandb: WandB run object
            bucket_name (str): Name of the GCS bucket to save checkpoints
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.checkpointpath = checkpointpath
        self.runname = runname

        # Initialize GCS client if bucket_name is provided
        if bucket_name is None:
            try:
                from utilities.dev_utils import get_gcs_bucket_name

                self.bucket_name = get_gcs_bucket_name()
            except (ImportError, ValueError) as e:
                print(f"Warning: Could not get bucket name from environment: {str(e)}")
                self.bucket_name = None
        else:
            self.bucket_name = bucket_name

        if self.bucket_name:
            try:
                self.storage_client = storage.Client()
                self.bucket = self.storage_clientcket(self.bucket_name)
            except Exception as e:
                self.bucket = None
        else:
            self.bucket = None

        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpointpath, exist_ok=True)

    def __call__(self, val_loss, model, epoch, optimizer=None, config=None, wandb=None):
        if self.patience > 0:
            score = -val_loss
            logger.set_context("OPTIMIZATION")
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_loss, model, epoch, optimizer, config, wandb)
            elif score < self.best_score + self.delta:
                self.counter += 1
                logger.info(
                    f"Validation Loss DOESN'T improve [{val_loss:.6f} >  {self.val_loss_min:.6f}] {self.counter} / {self.patience} epochs without improvements"
                )
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(val_loss, model, epoch, optimizer, config, wandb)
                self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch, optimizer=None, config=None, wandb=None):
        """Saves model when validation loss decrease."""
        if self.verbose:
            if epoch != 0:
                logger.info(
                    f"Validation Loss IMPROVES [{self.val_loss_min:.6f} --> {val_loss:.6f}], saving checkpoint",
                    end=" ",
                )

        self.val_loss_min = val_loss

        try:
            # Save complete checkpoint locally first
            checkpoint_path = os.path.join(self.checkpointpath, "weights_best.pt")
            model._forward_hooks.clear()

            # Create complete checkpoint with all required keys
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
                "config": config if config is not None else {},
                "wandb_run_id": getattr(wandb, 'id', None) if wandb is not None else None,
            }

            torch.save(checkpoint, checkpoint_path, pickle_module=dill)

            # Get run name for artifact path
            saveto_statedict = f"ARTIFACTS/{self.runname}.pt"

            # Upload to GCS if bucket is configured
            if self.bucket:
                logger.set_context("GCLOUD")
                try:
                    # Verify bucket access before attempting upload

                    blob = self.bucket.blob(saveto_statedict)
                    blob.upload_from_filename(checkpoint_path)
                    if self.verbose:
                        logger.info(
                            f"Saved model state dict to gs://{self.bucket_name}/{saveto_statedict}"
                        )
                    # Remove local file after successful upload
                    os.remove(checkpoint_path)
                except Exception as e:
                    logger.warning(f"Error saving to GCS: {str(e)}")
                    logger.warning("Debug info:")
                    logger.warning(f"- Bucket name: {self.bucket_name}")
                    logger.warning(f"- File path: {checkpoint_path}")
                    logger.warning(f"- Target path: {saveto_statedict}")
                    logger.warning(f"Checkpoint kept locally at: {checkpoint_path}")

                    # Log the full error details
                    import traceback

                    logger.warning("Full error traceback:")
                    logger.warning(traceback.format_exc())

        except Exception as e:
            logger.error(f"Error in save_checkpoint: {str(e)}")
            raise

    def load_checkpoint(self, model, checkpoint_path):
        """
        Load a checkpoint from GCS or local path

        Args:
            model: PyTorch model to load weights into
            checkpoint_path (str): Path to checkpoint (GCS path or local path)
        """
        try:
            if self.bucket and checkpoint_path.startswith("ARTIFACTS/"):
                # Download from GCS to temporary file
                tmp_path = os.path.join(self.checkpointpath, "temp_checkpoint.pt")
                blob = self.bucket.blob(checkpoint_path)
                blob.download_to_filename(tmp_path)
                checkpoint_path = tmp_path

            # Load state dict
            state_dict = torch.load(checkpoint_path)
            model.load_state_dict(state_dict)
            logger.set_context("GCLOUD")

            # Clean up temporary file if used
            if self.bucket and checkpoint_path.startswith(self.checkpointpath):
                os.remove(checkpoint_path)

            if self.verbose:
                logger.info(f"Loaded checkpoint from: {checkpoint_path}")

        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")

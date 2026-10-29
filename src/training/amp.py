import torch

class AMPManager:
    """
    Manages Automatic Mixed Precision (AMP) logic for PyTorch.
    Uses GradScaler for FP16 training to prevent underflow.
    """
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.device_type = "cuda" if torch.cuda.is_available() else "cpu"
        # GradScaler is only needed and supported for CUDA
        self.scaler = torch.cuda.amp.GradScaler() if (self.enabled and self.device_type == "cuda") else None

    def autocast(self):
        """Context manager for AMP."""
        if self.enabled and self.device_type == "cuda":
            return torch.amp.autocast(device_type=self.device_type, dtype=torch.float16)
        else:
            # Fallback for CPU or disabled AMP
            import contextlib
            return contextlib.nullcontext()

    def scale_loss(self, loss):
        """Scales the loss if GradScaler is active."""
        if self.scaler is not None:
            return self.scaler.scale(loss)
        return loss

    def step(self, optimizer):
        """Steps the optimizer and updates the scaler."""
        if self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()

    def state_dict(self):
        """Returns scaler state dict if active, otherwise None."""
        if self.scaler is not None:
            return self.scaler.state_dict()
        return None

    def load_state_dict(self, state_dict):
        """Restores scaler state dict if provided and scaler is active."""
        if self.scaler is not None and state_dict is not None:
            self.scaler.load_state_dict(state_dict)

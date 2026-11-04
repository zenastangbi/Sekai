"""Model and processor container."""

from torch import nn


class Route(nn.Module):
    """Container for a Qwen module and its processor."""

    def __init__(self, qwen, processor):
        """Register qwen as a child module and store processor."""
        super().__init__()
        self.qwen = qwen
        self.processor = processor

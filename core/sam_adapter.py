"""Tensor conversion for SAM prompt requests."""


class TensorPromptBackend:
    """SAM predictor wrapper for tensor-valued prompts."""

    def __init__(self, model):
        """Store the SAM predictor as model."""
        self.model = model

    def handle_request(self, request):
        """Return the SAM response to a tensor-converted request."""
        import torch

        r = dict(request)
        if r["type"] == "add_prompt":
            r["points"] = torch.tensor(r["points"], dtype=torch.float32)
            r["point_labels"] = torch.tensor(
                r["point_labels"], dtype=torch.int32
            )
            r.update(clear_old_points=True, output_prob_thresh=0.5)
        return self.model.handle_request(r)

    def propagate_in_video(self, **kwargs):
        """Return SAM propagation packets using kwargs."""
        return self.model.propagate_in_video(**kwargs)

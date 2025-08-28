import torch


class SafeParamHead(torch.nn.Module):
    def __init__(self, min_std: float = 1e-3, max_std: float = 5.0):
        super().__init__()
        self.min_std = min_std
        self.max_std = max_std

    def forward(self, x: torch.Tensor):
        # x: [..., 2 * action_dim]
        loc, raw = torch.chunk(x, 2, dim=-1)
        scale = torch.nn.functional.softplus(raw) + 1e-6  # >0
        # clamp to avoid extremes that can destabilize sampling
        scale = scale.clamp(self.min_std, self.max_std)

        # final NaN guards (very cheap)
        loc = torch.nan_to_num(loc, nan=0.0, posinf=10.0, neginf=-10.0)
        scale = torch.nan_to_num(
            scale, nan=1.0, posinf=self.max_std, neginf=self.min_std
        )
        return loc, scale

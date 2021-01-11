import torch
import torch.nn as nn

class EDMLoss(nn.Module):
    def __init__(self):
        super(EDMLoss, self).__init__()

    def forward(self, p_target: torch.Tensor, p_estimate: torch.Tensor):
        assert p_target.shape == p_estimate.shape
        # cdf for values [1, 2, ..., 10]
        cdf_target = torch.cumsum(p_target, dim=-1)
        # cdf for values [1, 2, ..., 10]
        cdf_estimate = torch.cumsum(p_estimate, dim=-1)
        cdf_diff = cdf_estimate - cdf_target
        emd = torch.sqrt(torch.mean(torch.pow(torch.abs(cdf_diff), 2), -1))
        return emd.mean()
        # samplewise_emd = torch.sqrt(torch.mean(torch.pow(torch.abs(cdf_diff), 2)))
        # return samplewise_emd.mean()

# np.abs(np.cumsum(y_true, axis=-1) - np.cumsum(y_pred, axis=-1))
# np.sqrt(np.mean(np.square(  ) ) )

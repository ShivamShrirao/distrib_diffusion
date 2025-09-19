import torch


class FlowScheduler:
    def __init__(self, num_steps, shift=1.0, device="cuda"):
        self.device = device
        self.setup(num_steps, shift)

    def setup(self, num_steps, shift=1.0):
        self.num_steps = num_steps
        self.sigmas = torch.linspace(1, num_steps, num_steps, device=self.device) / num_steps
        self.shift = shift
        self.sigmas = shift * self.sigmas / (1 + (shift - 1) * self.sigmas)

    def step(self, model, step_idx, x_t, cond=None, cfg=1):
        sigma_idx = len(self.sigmas)-step_idx-1
        cur_sigma = self.sigmas[sigma_idx]
        if sigma_idx-1 < 0:
            prev_sigma = 0.0
        else:
            prev_sigma = self.sigmas[sigma_idx-1]
        
        if cfg > 1:
            x_t = torch.cat([x_t] * 2)
            cond = torch.cat([torch.zeros_like(cond), cond])

        pred = model(x=x_t, t=cur_sigma.unsqueeze(0).repeat(x_t.shape[0], 1), c=cond)

        if cfg > 1:
            pred_uncond, pred_cond = pred.chunk(2)
            pred = pred_uncond + (pred_cond - pred_uncond) * cfg
            x_t, _ = x_t.chunk(2)

        dt = cur_sigma - prev_sigma
        x_prev = x_t - dt * pred
        return x_prev


class FlowMidPointScheduler(FlowScheduler):
    def step(self, model, step_idx, x_t, cond=None, cfg=1):
        sigma_idx = len(self.sigmas)-step_idx-1
        cur_sigma = self.sigmas[sigma_idx]
        if sigma_idx-1 < 0:
            prev_sigma = 0.0
        else:
            prev_sigma = self.sigmas[sigma_idx-1]
        mid_sigma = (cur_sigma + prev_sigma) / 2

        if cfg > 1:
            x_t = torch.cat([x_t] * 2)
            cond = torch.cat([torch.zeros_like(cond), cond])

        pred = model(x=x_t, t=cur_sigma.unsqueeze(0).repeat(x_t.shape[0], 1), c=cond)
        x_mid = x_t - (cur_sigma - mid_sigma) * pred
        pred_mid = model(x=x_mid, t=mid_sigma.unsqueeze(0).repeat(x_mid.shape[0], 1), c=cond)

        if cfg > 1:
            pred_uncond, pred_cond = pred_mid.chunk(2)
            pred_mid = pred_uncond + (pred_cond - pred_uncond) * cfg
            x_t, _ = x_t.chunk(2)

        x_prev = x_t - (cur_sigma - prev_sigma) * pred_mid
        return x_prev

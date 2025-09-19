import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from scipy.optimize import linear_sum_assignment
from sklearn.datasets import make_moons
from torch_linear_assignment import assignment_to_indices, batch_linear_assignment
from tqdm import tqdm

import src.models
import wandb
from src.metrics import AvgMeter, mmd_rbf, wasserstein_distance_2d
from src.schedulers import FlowMidPointScheduler, FlowScheduler

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def init_model(config):
    ModelClass = getattr(src.models, config.class_name)
    return ModelClass(**config.params)


def init_optimizer(config, model):
    optimizer_dict = {
        "adamw": torch.optim.AdamW,
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD,
    }
    return optimizer_dict[config.name](model.parameters(), **config.params)


@torch.no_grad()
def infer(
    config,
    model,
    scheduler,
    random_points,
    cond=None,
    cfg=1,
    idx=0,
):
    infer_steps = len(scheduler.sigmas)
    model.eval()
    x_t = random_points
    selected_set = set(np.unique(np.linspace(0, infer_steps - 1, 6, dtype=int)).tolist())
    assert infer_steps-1 in selected_set

    snapshots = []
    for step in range(infer_steps):
        x_t = scheduler.step(model, step, x_t, cond=cond, cfg=cfg)
        if step in selected_set:
            x_gen = x_t.cpu().numpy().copy()
            snapshots.append((step, x_gen))

    if snapshots:
        cols = len(snapshots)
    if config.wandb.enabled and idx % config.training.img_log_interval == 0:
        fig, axes = plt.subplots(1, cols, figsize=(6 * cols, 6), sharex=True, sharey=True)
        axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
        for ax, (s, pts) in zip(axes, snapshots):
            if config.class_conditional:
                c = cond.argmax(dim=1).cpu().numpy()
            else:
                c = None
            ax.scatter(pts[:, 0], pts[:, 1], c=c)
            ax.set_title(f"Step {s+1}")
            ax.set_xlim(-2, 3)
            ax.set_ylim(-2, 2)
            plt.tight_layout()
            wandb.log({f"FlowScheduler_steps:{infer_steps:02d}": wandb.Image(fig)}, step=idx)
            plt.close(fig)

    model.train()
    return x_t


def eval_metrics(eval_samples, gen_samples, prefix=""):
    metrics = {}

    w_dist = wasserstein_distance_2d(eval_samples, gen_samples)
    metrics[f'{prefix}wasserstein_distance'] = w_dist

    mmd = mmd_rbf(eval_samples, gen_samples)
    metrics[f'{prefix}mmd'] = mmd

    return metrics


@torch.no_grad()
def hungarian_match(real, noise):
    cost = torch.cdist(real, noise, p=2).pow(2)
    assignment = batch_linear_assignment(cost.unsqueeze(0))
    row_ind, col_ind = assignment_to_indices(assignment)
    # row_ind, col_ind = linear_sum_assignment(cost.cpu().numpy())
    return noise[col_ind.flatten()]


def main():
    config = OmegaConf.load("src/configs/train.yaml")
    cli_config = OmegaConf.from_cli()
    config = OmegaConf.merge(config, cli_config)

    seed_everything(config.seed)
    device = torch.device("cuda")

    if config.wandb.enabled:
        run = wandb.init(
            project="Distrib_Diffusion",
            config=OmegaConf.to_container(config, resolve=True),
        )

    model = init_model(config.model).to(device)
    opt = init_optimizer(config.optimizer, model)

    model.train()

    loss_log = AvgMeter()
    eval_samples, eval_labels = make_moons(n_samples=1000, noise=0.03, random_state=config.seed)
    eval_samples = torch.from_numpy(eval_samples).float().to(device, non_blocking=True)
    eval_labels = torch.from_numpy(eval_labels).to(device, non_blocking=True)
    eval_one_hot = F.one_hot(eval_labels, num_classes=2).float()
    random_points = torch.randn(1000, 2, device=device, generator=torch.Generator(device=device).manual_seed(config.seed))

    log = {}
    pbar = tqdm(range(1, config.training.train_steps+1))
    for i in pbar:
        x_0, y = make_moons(n_samples=config.training.batch_size, noise=0.03, random_state=i)
        x_0 = torch.from_numpy(x_0).float()
        if config.class_conditional:
            y = F.one_hot(torch.from_numpy(y), num_classes=2).float()
            y = y.to(device, non_blocking=True)
            p = torch.rand(y.shape[0], device=device)
            y[p <= 0.1] = 0         # for CFG
        else:
            y = None

        x_0 = x_0.to(device, non_blocking=True)
        noise = torch.randn_like(x_0)
        if config.optimal_transport_training:
            noise = hungarian_match(x_0, noise)
        t = torch.rand(x_0.shape[0], device=device).unsqueeze(1)
        x_t = (1 - t) * x_0 + t * noise
        model_pred = model(x_t, t, y)
        target = noise - x_0
        loss = F.mse_loss(model_pred, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        loss_log.update(loss.detach())

        if i % config.training.eval_interval == 0 or i == 1:
            scheduler = FlowScheduler(num_steps=1, shift=1.0, device=device)
            x_gen = infer(config=config, model=model, scheduler=scheduler, random_points=random_points, cond=eval_one_hot, cfg=config.cfg, idx=i)
            scheduler.setup(num_steps=2, shift=1.0)
            x_gen = infer(config=config, model=model, scheduler=scheduler, random_points=random_points, cond=eval_one_hot, cfg=config.cfg, idx=i)
            scheduler.setup(num_steps=5, shift=1.0)
            x_gen = infer(config=config, model=model, scheduler=scheduler, random_points=random_points, cond=eval_one_hot, cfg=config.cfg, idx=i)
            scheduler.setup(num_steps=10, shift=1.0)
            x_gen = infer(config=config, model=model, scheduler=scheduler, random_points=random_points, cond=eval_one_hot, cfg=config.cfg, idx=i)
            scheduler.setup(num_steps=20, shift=1.0)
            x_gen = infer(config=config, model=model, scheduler=scheduler, random_points=random_points, cond=eval_one_hot, cfg=config.cfg, idx=i)
            scheduler.setup(num_steps=50, shift=1.0)
            x_gen = infer(config=config, model=model, scheduler=scheduler, random_points=random_points, cond=eval_one_hot, cfg=config.cfg, idx=i)
            if config.class_conditional:
                metrics = {}
                for c in torch.unique(eval_labels):
                    metrics.update(eval_metrics(eval_samples[eval_labels == c], x_gen[eval_labels == c], prefix=f"class_{c}/"))
                avg_metrics = defaultdict(list)
                for k, v in metrics.items():
                    if "class_" in k:
                        avg_metrics[k.split("/")[1]].append(v)
                for k, v in avg_metrics.items():
                    avg_metrics[k] = sum(v) / len(v)
                metrics.update(avg_metrics)
            else:
                metrics = eval_metrics(eval_samples, x_gen)
            log.update(metrics)
            if config.wandb.enabled:
                wandb.log(log, step=i)

        if i % config.training.log_interval == 0 or i == 1:
            loss_dict = {"loss": loss_log.avg.item()}
            if config.wandb.enabled:
                wandb.log(loss_dict, step=i)
            pbar.set_postfix(log | loss_dict)
            loss_log.reset()


if __name__ == "__main__":
    main()

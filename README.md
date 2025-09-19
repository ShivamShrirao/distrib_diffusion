# Simple Diffusion model on two moons distribution

The goal is to train a diffusion model for 2D points to generate the two moons distribution.

![Two Moons Distribution](images/two_moons.png)

# Method

I follow the flow matching framework to build a probability path from a simple distribution (e.g., Gaussian) `p` to the target distribution (two moons) `q`. The model learns a time-dependent vector field that transports samples from `p` to `q` via a mean squared error objective between the predicted and true velocity fields.

We define a linear probability path where the sample x<sub>t</sub> at time `t` is:

x<sub>t</sub> = (1 - t) * x<sub>0</sub> + t * x<sub>1</sub>

with x<sub>0</sub> ~ `q` and x<sub>1</sub> ~ `p`. The target velocity field along this path is:

v(x<sub>t</sub>, t) = x<sub>1</sub> - x<sub>0</sub>

# Metrics

The metrics used to evaluate the model are:

- **Wasserstein Distance**: Computes the Earth Mover's Distance between normalized generated and target distributions using optimal transport. It measures the minimum cost to transform one distribution into another, where cost is based on Euclidean distance between sample points. Lower values indicate better distributional matching.

- **MMD (Maximum Mean Discrepancy)**: Uses an RBF (Radial Basis Function) kernel to compare two distributions by how similar their samples are. Lower is better.

# Experiment 1

We use an MLP that takes 2D points and a time step `t` and outputs a 2D vector. The architecture has 3 hidden layers of 128 units with SiLU activations. The time step `t` is concatenated to the input points for conditioning.

Training uses AdamW with lr=2e-3, batch size=256, weight decay=0.01, for 10,000 iterations.

### How to run
Install dependencies with `uv sync`, configure the settings in `src/configs/train.yaml`, and start training with `python train.py`.


Training logs: [https://wandb.ai/shivamshrirao/Distrib_Diffusion/runs/7j14hmty](https://wandb.ai/shivamshrirao/Distrib_Diffusion/runs/7j14hmty)

![Metrics](images/Experiment_1_metrics.png)

| Metric | Best Value |
|--------|------------|
| **Wasserstein Distance** ↓ | 0.086 |
| **MMD** ↓ | 0.000467 |


Within 4,000 iterations, the model generates samples that closely resemble the two moons distribution (inference with 50 steps):
![Generated Samples 4000](images/media_images_FlowScheduler_steps_50_4000_908ce1792dbc7cf92c88.png)

After 10,000 iterations, the outputs are cleaner:
![Generated Samples 10000](images/media_images_FlowScheduler_steps_50_10000_1205693cb2350ffd003c.png)

# Experiment 2

I explored sinusoidal positional embeddings for `t` instead of concatenation.

Training logs: [https://wandb.ai/shivamshrirao/Distrib_Diffusion/runs/19s8znjy](https://wandb.ai/shivamshrirao/Distrib_Diffusion/runs/19s8znjy)

![Metrics](images/Experiment_2_metrics.png)

| Metric | Best Value |
|--------|------------|
| **Wasserstein Distance** ↓ | 0.091 |
| **MMD** ↓ | 0.000526 |


Results are similar across training iterations:
![Generated Samples 10000 Sinusoidal](images/media_images_FlowScheduler_steps_50_10000_6e7c79790cec4f991f48.png)

Given the simplicity of the task, just concatenation of `t` is sufficient.

# Experiment 3

To further improve quality, I used optimal transport (OT) to initialize samples from the Gaussian `p` closer to the target `q`. Practically, this reorders batched samples from `p` to be matched with nearby samples from `q`, shortening transport distances along the path.

Training logs: [https://wandb.ai/shivamshrirao/Distrib_Diffusion/runs/cau5bhft](https://wandb.ai/shivamshrirao/Distrib_Diffusion/runs/cau5bhft)

![Metrics](images/Experiment_3_metrics.png)

| Metric | Best Value |
|--------|------------|
| **Wasserstein Distance** ↓ | 0.064 |
| **MMD** ↓ | 0.000150 |


Even after just 2000 iterations, the model produces high-quality samples.

| Iterations | **Experiment 1**                                                                                      | **Experiment 3 (Optimal Transport)**                                                                     |
| ---------- | ------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------- |
| **2000**   | ![Generated Samples 2000](images/media_images_FlowScheduler_steps_50_2000_49999de194c5bfae9721.png)   | ![Generated Samples 2000 OT](images/media_images_FlowScheduler_steps_50_2000_ea36aeb3f1a7dda12afa.png)  |
| **4000**   | ![Generated Samples 4000](images/media_images_FlowScheduler_steps_50_4000_908ce1792dbc7cf92c88.png)   | ![Generated Samples 4000 OT](images/media_images_FlowScheduler_steps_50_4000_589a7f0c3f2bf35c879c.png)  |
| **10000**  | ![Generated Samples 10000](images/media_images_FlowScheduler_steps_50_10000_1205693cb2350ffd003c.png) | ![Generated Samples 10000 OT](images/media_images_FlowScheduler_steps_50_10000_44c21fd9290d5ac8093e.png) |

Inference used 50 steps.

With OT, training converges faster and samples are higher quality. The inference trajectory is cleaner and more stable, and the two-moons structure can be seen forming earlier:
![Inference Trajectory](images/media_images_FlowScheduler_steps_50_10000_44c21fd9290d5ac8093e.png)

This also enables good samples in fewer inference steps. Below compares outputs at different inference step counts:

| Infer Steps | **Experiment 1**                                                                                  | **Experiment 3 (Optimal Transport)**                                                                 |
| ----------- | -------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| **10**      | ![Generated Samples 10](images/media_images_FlowScheduler_steps_10_10000_76b10b84062e68d3a904.png) | ![Generated Samples 10 OT](images/media_images_FlowScheduler_steps_10_10000_6f4a1e39450fee01d4ce.png) |
| **5**       | ![Generated Samples 5](images/media_images_FlowScheduler_steps_5_10000_bffe83751bf65c6c8caf.png)   | ![Generated Samples 5 OT](images/media_images_FlowScheduler_steps_5_10000_c407795028026f12cc56.png)   |
| **2**       | ![Generated Samples 2](images/media_images_FlowScheduler_steps_2_10000_a340c2e9b8916165b085.png)   | ![Generated Samples 2 OT](images/media_images_FlowScheduler_steps_2_10000_a2e8346d35c25ad7a43f.png)   |
| **1**       | ![Generated Samples 1](images/media_images_FlowScheduler_steps_1_9900_efecec1af5e6582d2f3d.png)    | ![Generated Samples 1 OT](images/media_images_FlowScheduler_steps_1_9900_c54e5e03d57025062215.png)    |

As seen above, OT yields good quality with far fewer inference steps, even a single step produces samples resembling two moons.

# Experiment 4

Since `make_moons` provides class labels, we can train a class-conditional model and apply classifier-free guidance (CFG) at inference.

For conditioning, I one-hot encode the class labels and concatenate them to the input along with time `t`. The rest follows Experiment 3.

Training logs: [https://wandb.ai/shivamshrirao/Distrib_Diffusion/runs/5umtltkm](https://wandb.ai/shivamshrirao/Distrib_Diffusion/runs/5umtltkm)

![Metrics](images/Experiment_4_metrics.png)

| Metric | Best Value |
|--------|------------|
| **MMD** ↓ | 0.030 |
| **Wasserstein Distance** ↓ | 0.484 |

With 10 inference steps:

| Training Iterations | **CFG = 1**                                                                                 | **CFG = 3**                                                                                 |
| ------------------- | -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| **2000**            | ![Generated Samples CFG 1, 2000](images/media_images_FlowScheduler_steps_10_2000_b15fdb29822b0397fa57.png) | ![Generated Samples CFG 3, 2000](images/media_images_FlowScheduler_steps_10_2000_6ff7b002cb0832920634.png) |
| **6000**            | ![Generated Samples CFG 1, 6000](images/media_images_FlowScheduler_steps_10_6000_e6108afc4765a16cb866.png) | ![Generated Samples CFG 3, 6000](images/media_images_FlowScheduler_steps_10_6000_e683bccfd383077f938f.png) |

Class conditioning plus CFG sharpens structure and alignment to the target distribution. Although it can come at the cost of variance.

Comparing the effect of OT at 10 inference steps, CFG=3:

| **Model Type**              | **Generated Samples**                                                                 |
| --------------------------- | ------------------------------------------------------------------------------------- |
| **Without Optimal Transport** | ![Generated Samples Without OT](images/media_images_FlowScheduler_steps_10_10000_9e11ad5287770282ecc8.png) |
| **With Optimal Transport**    | ![Generated Samples With OT](images/media_images_FlowScheduler_steps_10_10000_6ac187cc4d76e154ede5.png)     |

With OT, trajectories are straighter and avoid back-and-forth motion before settling into the target distribution.

# Some more experiments
- Using FlowMidPointScheduler which does two function evaluations per step, one at the midpoint to more accurately estimate the trajectory. The outputs and metrics turned out to be very similar to using the regular FlowScheduler for twice the number of steps.
- Using shift to change if model spends more time at the start or end of the trajectory. For simple distributions like two moons, this did not make much difference in metrics.
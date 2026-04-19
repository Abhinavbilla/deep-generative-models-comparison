# Deep Generative Models Comparison: VAE, InfoVAE, and VQ-VAE

This repository contains a comprehensive empirical comparison of continuous and discrete latent space constraints in deep generative models. We analyze Standard VAEs, InfoVAEs, and VQ-VAEs to evaluate their reconstruction accuracy, generative realism, and latent space interpolation, with a specific deep-dive into the mechanics of **posterior collapse**.

## Project Overview
The structure of a model's latent space directly determines its generative capabilities. In this project, we trained and evaluated three distinct architectures on the **CelebA** dataset:
1. **Standard VAE:** Uses a continuous Gaussian latent space with KL regularization.
2. **InfoVAE:** Modifies the ELBO objective to use Maximum Mean Discrepancy (MMD) on the aggregated posterior, decoupling inference from regularization.
3. **VQ-VAE:** Replaces the continuous bottleneck with a discrete, finite codebook of embeddings, mapping encoded inputs via nearest-neighbor lookup.

## Key Findings & Results

### 1. Generative Quality (FID) and Reconstruction (MSE)
Discrete latent representations structurally eliminate the Gaussian smoothing problem inherent to standard VAEs. VQ-VAE produced the sharpest and most realistic outputs.

| Model | Reconstruction Loss (MSE) | Fréchet Inception Distance (FID) |
| :--- | :---: | :---: |
| **Standard VAE** | 0.0337 | 85.29 |
| **InfoVAE** | 0.0756 | 68.21 |
| **VQ-VAE** | **0.0028** | **18.18** |

### 2. Posterior Collapse Ablation Study
We investigated why highly capable decoders ignore the latent space. By tracking "active latent units" (dimensions where variance $> 0.01$), we proved that posterior collapse is an objective-driven problem, not an architectural one.
* **Standard Regularization ($\beta=1.0$):** Resulted in 0/32 active units (total collapse).
* **Weakened Regularization ($\beta=0.00025$):** Restored 32/32 active units (100% utilization). 
* **Depth Alteration:** Increasing decoder depth from 3 to 5 layers had zero effect on preventing collapse.
* **Objective Fix:** InfoVAE modifies the objective to introduce the MMD term to ensure proper latent space structuring, which prevents collapse
* **Structural Fix:** VQ-VAE structurally guarantees active latent utilization by forcing nearest-neighbor codebook lookups, entirely avoiding collapse.

### 3. Latent Space Interpolation (LPIPS)
We measured the smoothness and semantic coherence of linear interpolations between distinct data points.

| Model | LPIPS Score | Observation |
| :--- | :---: | :---: |
| **Standard VAE** | 0.3106 | Continuous transitions, but blurry with lost semantic detail. |
| **InfoVAE** | 0.1832 | Sharper transitions, but abrupt changes between intermediate frames. |
| **VQ-VAE** | **0.0155** | Smoothest interpolation with highly coherent semantic transitions. |

##  Repository Structure
* `docs/`: Project documentation, final reports, and presentation slides.
* `models/`: Core PyTorch architectural implementations for the Standard VAE, InfoVAE, and VQ-VAE.
* `papers/`: Reference research literature and foundational papers (e.g., Kingma & Welling, van den Oord et al., Zhao et al.).
* `scripts/`: Training scripts, evaluation loops (MSE, FID, LPIPS), and notebooks used for the ablation studies.

##  Contributors
* Abhinav
* Devi Prasad
* Dhanunjaya
* Lokesh
* Swarup

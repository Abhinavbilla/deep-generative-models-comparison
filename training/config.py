class Config:
    # =========================
    # General
    # =========================
    model_name = "vqvae"   # "vae" or "infovae"
    device = "cuda"

    # =========================
    # Dataset
    # =========================
    image_size = 64
    batch_size = 64
    num_workers = 0

    # =========================
    # Training
    # =========================
    epochs = 50
    lr = 3e-4
    log_interval = 100

    # =========================
    # VQ-VAE params
    # =========================
    embed_dim = 64
    num_embeddings = 256
    beta = 0.25

    # =========================
    # Saving
    # =========================
    save_dir = "trained_models"
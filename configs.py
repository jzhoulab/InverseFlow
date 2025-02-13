default_config = {
    "modelstr": "GaussianICM",
    "checkpoint_path": "./checkpoints",
    "log_path": "./logs",
    "device": "cuda",
    "train": {
        "num_timesteps": 11,
        "batch_size": 32,
        "lr": 1e-4,
        "n_epochs": 2000,
        "schedule": {
            "t_min": 0.002,
            "t_max": 0.1,
            "t_data": 0.5,
            "rho": 7.0,
        },
        "save_interval": 100,
    },
    "flow": {
        "type": "Gaussian",
        "sigma": 25 / 255.0,
        "algorithm": "icm",
    },
    "model": {
        "name": "N2NUNet",
        "input_channel": 1,
        "input_size": 256,
        "channels": [128, 128, 256, 256, 512],
        "embed_dim": 512,
        "embed_scale": 1.0,
    },
    "dataset": {
        "file_path": "/path/to/dataset/train_noisy_gray_l.npz",
        "gt_tag": "gray_imgs",
        "noisy_tag": "noisy_gray_imgs",
    },
}

gaussian_icm_config = default_config.copy()
gaussian_icm_config.update(
    {
        "modelstr": "GaussianICM",
        "flow": {
            "type": "Gaussian",
            "sigma": 25 / 255.0,
            "algorithm": "icm",
        },
        "dataset": {
            "file_path": "/path/to/dataset/train_noisy_gray_l.npz",
            "gt_tag": "gray_imgs",
            "noisy_tag": "noisy_gray_imgs",
        },
    }
)

gaussian_ifm_config = default_config.copy()
gaussian_ifm_config.update(
    {
        "modelstr": "GaussianIFM",
        "flow": {
            "type": "Gaussian",
            "sigma": 25 / 255.0,
            "algorithm": "ifm",
        },
        "dataset": {
            "file_path": "/path/to/dataset/train_noisy_gray_l.npz",
            "gt_tag": "gray_imgs",
            "noisy_tag": "noisy_gray_imgs",
        },
    }
)

poisson_icm_config = default_config.copy()
poisson_icm_config.update(
    {
        "modelstr": "PoissonICM",
        "flow": {
            "type": "Poisson",
            "k": 0.01,
            "algorithm": "icm",
        },
        "dataset": {
            "file_path": "/path/to/dataset/train_poisson_noisy_gray_l_001.npz",
            "gt_tag": "gray_imgs",
            "noisy_tag": "poisson_noisy_imgs",
        },
    }
)


jacobi_icm_config = default_config.copy()
jacobi_icm_config.update(
    {
        "modelstr": "JacobiICM",
        "flow": {
            "type": "Jacobi",
            "alpha": 1.0,
            "beta": 1.0,
            "tmax": 0.2,
            "algorithm": "icm",
        },
        "dataset": {
            "file_path": "/path/to/dataset/train_DDSM_noisy_gray_l.npz",
            "gt_tag": "gray_imgs",
            "noisy_tag": "DDSM_noisy_imgs",
        },
    }
)


g2_icm_config = default_config.copy()
g2_icm_config.update(
    {
        "modelstr": "GaussianG2ICM",
        "flow": {
            "type": "GaussianG2",
            "sigma": 25 / 255.0,
            "span": 2,
            "algorithm": "icm",
        },
        "dataset": {
            "file_path": "/path/to/dataset/train_g2_noisy_gray_l.npz",
            "gt_tag": "gray_imgs",
            "noisy_tag": "g2_noisy_imgs",
        },
    }
)

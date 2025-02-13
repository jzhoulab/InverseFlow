import math
import numpy as np
import torch
from torch import Tensor
import os
from tqdm import tqdm
import logging
from model import ConsistencyModel
import inverse_flow
from utils import get_logger


def karras_schedule(
    num_timesteps: int,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    device: torch.device = None,
) -> Tensor:
    """Implements the karras schedule that controls the standard deviation of
    noise added.

    Parameters
    ----------
    num_timesteps : int
        Number of timesteps at the current point in training.
    sigma_min : float, default=0.002
        Minimum standard deviation.
    sigma_max : float, default=80.0
        Maximum standard deviation
    rho : float, default=7.0
        Schedule hyper-parameter.
    device : torch.device, default=None
        Device to generate the schedule/sigmas/boundaries/ts on.

    Returns
    -------
    Tensor
        Generated schedule/sigmas/boundaries/ts.
    """
    rho_inv = 1.0 / rho
    # Clamp steps to 1 so that we don't get nans
    steps = torch.arange(num_timesteps, device=device) / max(num_timesteps - 1, 1)
    sigmas = sigma_min**rho_inv + steps * (sigma_max**rho_inv - sigma_min**rho_inv)
    sigmas = sigmas**rho

    return sigmas


def createModel(config):
    model_name = config["model"]["name"]
    if hasattr(model, model_name):
        Model = getattr(model, model_name)
    else:
        raise NotImplementedError(f"Model {model_name} not implemented")
    base_model = Model(
        input_channel=config["model"]["input_channel"],
        channels=config["model"]["channels"],
        embed_dim=config["model"]["embed_dim"],
        embed_scale=config["model"]["embed_scale"],
    )
    if config["flow"]["algorithm"] == "icm":
        model = torch.nn.DataParallel(
            ConsistencyModel(
                base_model,
                sigma_data=config["train"]["schedule"]["t_data"],
                sigma_min=config["train"]["schedule"]["t_min"],
            )
        )
    elif config["flow"]["algorithm"] == "ifm":
        model = torch.nn.DataParallel(base_model)
    else:
        raise NotImplementedError("Algorithm not implemented")
    model = model.cuda()

    return model


def createFlow(config):
    class_name = config["flow"]["type"] + "Flow"
    if hasattr(inverse_flow, class_name):
        Flow = getattr(inverse_flow, class_name)
    else:
        raise NotImplementedError(f"Flow type {config['flow']['type']} not implemented")
    flow = Flow(config["flow"])

    return flow


class TrainLoop:
    def __init__(self, model, flow, data, config, val_data=None):
        self.model = model
        self.flow = flow
        self.data = data
        self.val_data = val_data
        self.n_epochs = config["train"]["n_epochs"]
        self.num_timesteps = config["train"]["num_timesteps"]
        self.schedule_param = config["train"]["schedule"]
        self.lr = config["train"]["lr"]
        self.modelstr = config["modelstr"]
        self.checkpoint_path = config["checkpoint_path"]
        self.save_interval = config["train"]["save_interval"]
        self.log_path = config["log_path"]

        os.makedirs(self.checkpoint_path, exist_ok=True)
        os.makedirs(self.log_path, exist_ok=True)

        self.logger = get_logger(os.path.join(self.log_path, f"{self.modelstr}.log"))

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

        self.losses = []
        self.device = config["device"]

        if self.val_data is None:
            self.val_data = self.data

    def train(self):
        self.model.train()  # Start the training loop
        current_training_step = 0
        for epoch in range(self.n_epochs):
            for x_noisy, _ in self.data:
                self.step(x_noisy)
                current_training_step += 1
            self.evaluate(epoch)
            if epoch % self.save_interval == 0:
                self.save(epoch)
            self.losses = []

    def step(self, x_noisy):
        x_noisy = x_noisy.float().to(self.device)
        ts = karras_schedule(
            self.num_timesteps,
            self.schedule_param["t_min"],
            self.schedule_param["t_max"],
            self.schedule_param["rho"],
            x_noisy.device,
        )
        with torch.no_grad():
            x_clean = self.flow.backward(x_noisy, self.model, ts)

        self.optimizer.zero_grad()
        x_t = self.flow.forward(x_clean, ts)
        timesteps = torch.randint(0, ts.shape[0] - 1, (x_clean.shape[0],))

        next_noisy_x = x_t[timesteps + 1, torch.arange(x_t.shape[1])]
        current_noisy_x = x_t[timesteps, torch.arange(x_t.shape[1])]

        loss = self.flow.compute_loss(
            current_noisy_x, next_noisy_x, self.model, ts, timesteps
        )
        loss.backward()
        self.optimizer.step()
        self.losses.extend(
            [loss.item()] * x_noisy.shape[0]
        )  # Store the loss for analysis

    def evaluate(self, epoch):
        # Implement evaluation logic here
        self.model.eval()
        ts = karras_schedule(
            self.num_timesteps,
            self.schedule_param["t_min"],
            self.schedule_param["t_max"],
            self.schedule_param["rho"],
            self.device,
        )
        val_psnrs = []
        with torch.no_grad():
            for x_noisy, x_gt in self.val_data:
                x_noisy = x_noisy.float().to(self.device)
                x_clean = self.flow.backward(x_noisy, self.model, ts)
                true_loss = torch.mean(
                    ((x_clean - x_gt.float().to(self.device)) ** 2),
                    dim=(tuple(range(1, x_gt.ndim))),
                )
                psnr = (-10 * torch.log10(true_loss)).mean().item()
                val_psnrs.append(psnr)

        self.model.train()
        logging.info(
            f"Epoch {epoch}: Avg Loss: {np.mean(self.losses)}; Validation PSNR: {np.mean(val_psnrs)}"
        )

    def predict(self, x_noisy):
        self.model.eval()
        ts = karras_schedule(
            self.num_timesteps,
            self.schedule_param["t_min"],
            self.schedule_param["t_max"],
            self.schedule_param["rho"],
            self.device,
        )
        with torch.no_grad():
            x_noisy = x_noisy.float().to(self.device)
            x_clean = self.flow.backward(x_noisy, self.model, ts)
        return x_clean

    def save(self, epoch):
        torch.save(
            self.model.state_dict(),
            os.path.join(self.checkpoint_path, f"{self.modelstr}.epoch{epoch}.pth"),
        )

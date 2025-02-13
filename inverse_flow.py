import math
import numpy as np

import torch
from torch import Tensor, nn
import torch.distributions.gamma as Gamma

from utils import pad_dims_like


def improved_loss_weighting(sigmas: Tensor) -> Tensor:
    """Computes the weighting for the consistency loss.

    Parameters
    ----------
    sigmas : Tensor
        Standard deviations of the noise.

    Returns
    -------
    Tensor
        Weighting for the consistency loss.

    References
    ----------
    [1] [Improved Techniques For Consistency Training](https://arxiv.org/pdf/2310.14189.pdf)
    """
    return 1 / (sigmas[1:] - sigmas[:-1])


def pseudo_huber_loss(input: Tensor, target: Tensor) -> Tensor:
    """Computes the pseudo huber loss.

    Parameters
    ----------
    input : Tensor
        Input tensor.
    target : Tensor
        Target tensor.

    Returns
    -------
    Tensor
        Pseudo huber loss.
    """
    c = 0.00054 * math.sqrt(math.prod(input.shape[1:]))
    return torch.sqrt((input - target) ** 2 + c**2) - c


def reverse_ODE_solver(model, x0, ts):
    x = x0
    rts = torch.flip(ts, [0]).cuda()
    steps = rts[:-1] - rts[1:]
    for i in range(steps.shape[0]):
        flow = model(x, torch.ones(x.shape[0], device=x.device) * rts[i])
        x = x - flow * steps[i]
    return x


class InverseFlow:
    """
    Implements the abstract Inverse Flow algorithm.
    """

    def __init__(self, algorithm: str = "icm"):
        self.algorithm = algorithm
        assert algorithm in ["icm", "ifm"], "Algorithm must be either icm or ifm"

    def sample(self, x: Tensor) -> Tensor:
        pass

    def forward(self, x: Tensor, ts: Tensor) -> Tensor:
        xt = self.sample(x)
        return x.unsqueeze(0) + pad_dims_like(ts / ts.max(), x).unsqueeze(1) * (
            xt - x
        ).unsqueeze(0)

    def backward(self, x: Tensor, model: nn.Module, ts: Tensor) -> Tensor:
        if self.algorithm == "icm":
            return model(x, torch.ones(x.shape[0], device=x.device) * ts[-1])
        elif self.algorithm == "ifm":
            return reverse_ODE_solver(model, x, ts)

    def compute_loss(
        self,
        current_noisy_x: Tensor,
        next_noisy_x: Tensor,
        model: nn.Module,
        ts: Tensor,
        timesteps: Tensor,
    ) -> Tensor:
        if self.algorithm == "icm":
            return self.compute_icm_loss(
                current_noisy_x, next_noisy_x, model, ts, timesteps
            )
        elif self.algorithm == "ifm":
            return self.compute_ifm_loss(
                current_noisy_x, next_noisy_x, model, ts, timesteps
            )

    def compute_icm_loss(
        self,
        current_noisy_x: Tensor,
        next_noisy_x: Tensor,
        model: nn.Module,
        ts: Tensor,
        timesteps: Tensor,
    ) -> Tensor:
        next_t = ts[timesteps + 1]
        current_t = ts[timesteps]
        next_x = model(next_noisy_x, next_t)
        with torch.no_grad():
            current_x = model(current_noisy_x, current_t)
        loss_weights = pad_dims_like(improved_loss_weighting(ts)[timesteps], next_x)
        return (pseudo_huber_loss(current_x, next_x) * loss_weights).mean()

    def compute_ifm_loss(
        self,
        current_noisy_x: Tensor,
        next_noisy_x: Tensor,
        model: nn.Module,
        ts: Tensor,
        timesteps: Tensor,
    ) -> Tensor:
        next_t = ts[timesteps + 1]
        current_t = ts[timesteps]
        flow = model(next_noisy_x, next_t)
        return pseudo_huber_loss(
            flow,
            (next_noisy_x - current_noisy_x)
            / pad_dims_like(next_t - current_t, current_noisy_x),
        ).mean()


class GaussianFlow(InverseFlow):
    """
    Implements the Gaussian Flow.
    """

    def __init__(self, config: dict):
        super().__init__(config["algorithm"])
        self.sigma = float(config["sigma"])

    def sample(self, x: Tensor) -> Tensor:
        return x + self.sigma * torch.randn_like(x)


class GaussianG2Flow(InverseFlow):
    """
    Implements the Gaussian G2 Flow.
    """

    def __init__(self, config: dict):
        super().__init__(config["algorithm"])
        self.sigma = float(config["sigma"])
        self.span = float(config["span"])

    def sample(self, x: Tensor) -> Tensor:
        h, w = x.shape[-2:]
        xx = torch.linspace(-h // 2, h // 2, h, device=x.device)
        yy = torch.linspace(-w // 2, w // 2, w, device=x.device)
        xx, yy = torch.meshgrid(xx, yy)

        r_squared = xx**2 + yy**2
        cos_term = torch.cos(torch.sqrt(r_squared))
        gaussian_pdf = (1 / (2 * torch.pi * self.span**2)) * torch.exp(
            -r_squared / (2 * self.span**2)
        )
        g2 = cos_term * gaussian_pdf
        g2 = g2 / (g2**2).sum() ** 0.5

        ori_noise = torch.randn_like(x) * self.sigma
        f_noise = torch.fft.fft2(ori_noise)
        f_g2 = torch.fft.fft2(g2)
        noise = torch.fft.ifft2(f_noise * f_g2).real

        return x + noise


class PoissonFlow(InverseFlow):
    """
    Implements the Poisson Flow.
    """

    def __init__(self, config: dict):
        super().__init__(config["algorithm"])
        self.k = config["k"]

    def sample(self, x: Tensor) -> Tensor:
        # Add Poisson noise with scaling parameter k
        poisson_noise = torch.poisson(x / self.k) * self.k
        return poisson_noise

    def backward(self, x: Tensor, model: nn.Module, ts: Tensor) -> Tensor:
        x = super().backward(x, model, ts)
        x = torch.clamp(x, 0, 1)
        return x

    def compute_icm_loss(
        self,
        current_noisy_x: Tensor,
        next_noisy_x: Tensor,
        model: nn.Module,
        ts: Tensor,
        timesteps: Tensor,
    ) -> Tensor:
        next_t = ts[timesteps + 1]
        current_t = ts[timesteps]
        next_x = model(next_noisy_x, next_t)
        next_x = torch.clamp(next_x, 0, 1)
        with torch.no_grad():
            current_x = model(current_noisy_x, current_t)
            current_x = torch.clamp(current_x, 0, 1)
        loss_weights = pad_dims_like(improved_loss_weighting(ts)[timesteps], next_x)
        return (pseudo_huber_loss(current_x, next_x) * loss_weights).mean()


class GammaFlow(InverseFlow):
    """
    Implements the Gamma Flow.
    """

    def __init__(self, config: dict):
        super().__init__(config["algorithm"])
        self.alpha = self.beta = 1 / float(config["sigma"]) ** 2

    def sample(self, x: Tensor):
        gamma = Gamma.Gamma(self.alpha, self.beta).sample(x.shape).to(x.device)
        return x * gamma


class JacobiFlow(InverseFlow):
    """
    Implements the Jacobi Flow.
    """

    def __init__(
        self,
        config: dict,
    ):
        super().__init__(config["algorithm"])
        self.a = config["alpha"]
        self.b = config["beta"]
        self.t = config["tmax"]

    def sample(self, x: Tensor, step_size=5e-4, eps=1e-5):
        """
        Generate Jacobi diffusion samples with the Euler-Maruyama solver.
        """

        def step(x, step_size):
            g = torch.sqrt(x * (1 - x))
            return (
                x
                + 0.5 * (self.a * (1 - x) - self.b * x) * step_size
                + torch.sqrt(step_size) * g * torch.randn_like(x)
            )

        time_steps = torch.linspace(0, self.t, int(self.t / step_size), device=x.device)
        step_size = time_steps[1] - time_steps[0]
        x_t = x.clone()
        with torch.no_grad():
            for time_step in time_steps:
                x_next = step(x_t, step_size)
                x_next = x_next.clip(eps, 1 - eps)
                x_t = x_next
        return x_t

    def backward(self, x: Tensor, model: nn.Module, ts: Tensor) -> Tensor:
        x = super().backward(torch.logit(x, eps=1e-5), model, ts)
        x = torch.sigmoid(x)
        return x

    def compute_icm_loss(
        self,
        current_noisy_x: Tensor,
        next_noisy_x: Tensor,
        model: nn.Module,
        ts: Tensor,
        timesteps: Tensor,
    ) -> Tensor:
        next_t = ts[timesteps + 1]
        current_t = ts[timesteps]
        next_x = model(torch.logit(next_noisy_x, eps=1e-5), next_t)
        next_x = torch.sigmoid(next_x)
        with torch.no_grad():
            current_x = model(torch.logit(current_noisy_x, eps=1e-5), current_t)
            current_x = torch.sigmoid(current_x)
        loss_weights = pad_dims_like(improved_loss_weighting(ts)[timesteps], next_x)
        return (pseudo_huber_loss(current_x, next_x) * loss_weights).mean()


class RayleighFlow(InverseFlow):
    """
    Implements the Rayleigh Flow.
    """

    def __init__(self, config: dict):
        super().__init__(config["algorithm"])
        self.ray_parameter = config["parameter"]

    def sample(self, x: Tensor):
        ray_noise = (
            torch.tensor(np.random.rayleigh(self.ray_parameter, x.shape)).float().cuda()
        )
        return x * (ray_noise + 1)

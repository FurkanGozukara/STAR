# Copyright (c) Alibaba, Inc. and its affiliates.

import torch
import torchsde
from tqdm.auto import trange

from video_to_video.utils.logger import get_logger

logger = get_logger()

def get_ancestral_step(sigma_from, sigma_to, eta=1.):
    """
    Calculates the noise level (sigma_down) to step down to and the amount
    of noise to add (sigma_up) when doing an ancestral sampling step.
    """
    if not eta:
        return sigma_to, 0.
    sigma_up = min(
        sigma_to,
        eta * (
            sigma_to**2 *  # noqa
            (sigma_from**2 - sigma_to**2) / sigma_from**2)**0.5)
    sigma_down = (sigma_to**2 - sigma_up**2)**0.5
    return sigma_down, sigma_up


def get_scalings(sigma):
    c_out = -sigma
    c_in = 1 / (sigma**2 + 1.**2)**0.5
    return c_out, c_in


@torch.no_grad()
def sample_heun(noise,
                model,
                sigmas,
                s_churn=0.,
                s_tmin=0.,
                s_tmax=float('inf'),
                s_noise=1.,
                show_progress=True,
                seed=None,
                **kwargs):
    """
    Implements Algorithm 2 (Heun steps) from Karras et al. (2022).
    """
    # Set up generator for deterministic noise if seed is provided
    generator = torch.Generator(device=noise.device)
    if seed is not None and seed >= 0:
        generator.manual_seed(seed)
    
    x = noise * sigmas[0]
    for i in trange(len(sigmas) - 1, disable=not show_progress):
        # Check for cancellation at the start of each diffusion step
        try:
            from logic.cancellation_manager import cancellation_manager
            cancellation_manager.check_cancel()
        except ImportError:
            # If cancellation manager is not available, continue without cancellation
            pass
        except Exception as e:
            # If there's a CancelledError or other cancellation exception, re-raise it
            if "cancel" in str(e).lower() or "cancelled" in str(e).lower():
                raise e
            # For other exceptions, continue without cancellation
            pass
            
        gamma = 0.
        if s_tmin <= sigmas[i] <= s_tmax and sigmas[i] < float('inf'):
            gamma = min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
        eps = torch.randn_like(x, generator=generator if seed is not None and seed >= 0 else None) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat**2 - sigmas[i]**2)**0.5
        if sigmas[i] == float('inf'):
            # Euler method
            denoised = model(noise, sigma_hat)
            x = denoised + sigmas[i + 1] * (gamma + 1) * noise
        else:
            _, c_in = get_scalings(sigma_hat)
            denoised = model(x * c_in, sigma_hat)
            d = (x - denoised) / sigma_hat
            dt = sigmas[i + 1] - sigma_hat
            if sigmas[i + 1] == 0:
                # Euler method
                x = x + d * dt
            else:
                # Heun's method
                x_2 = x + d * dt
                _, c_in = get_scalings(sigmas[i + 1])
                denoised_2 = model(x_2 * c_in, sigmas[i + 1])
                d_2 = (x_2 - denoised_2) / sigmas[i + 1]
                d_prime = (d + d_2) / 2
                x = x + d_prime * dt
    return x


class BatchedBrownianTree:
    """
    A wrapper around torchsde.BrownianTree that enables batches of entropy.
    """

    def __init__(self, x, t0, t1, seed=None, **kwargs):
        t0, t1, self.sign = self.sort(t0, t1)
        w0 = kwargs.get('w0', torch.zeros_like(x))
        if seed is None:
            seed = torch.randint(0, 2**63 - 1, []).item()
        self.batched = True
        try:
            assert len(seed) == x.shape[0]
            w0 = w0[0]
        except TypeError:
            seed = [seed]
            self.batched = False
        self.trees = [
            torchsde.BrownianTree(t0, w0, t1, entropy=s, **kwargs)
            for s in seed
        ]

    @staticmethod
    def sort(a, b):
        return (a, b, 1) if a < b else (b, a, -1)

    def __call__(self, t0, t1):
        t0, t1, sign = self.sort(t0, t1)
        w = torch.stack([tree(t0, t1) for tree in self.trees]) * (
            self.sign * sign)
        return w if self.batched else w[0]


class BrownianTreeNoiseSampler:
    """
    A noise sampler backed by a torchsde.BrownianTree.

    Args:
        x (Tensor): The tensor whose shape, device and dtype to use to generate
            random samples.
        sigma_min (float): The low end of the valid interval.
        sigma_max (float): The high end of the valid interval.
        seed (int or List[int]): The random seed. If a list of seeds is
            supplied instead of a single integer, then the noise sampler will
            use one BrownianTree per batch item, each with its own seed.
        transform (callable): A function that maps sigma to the sampler's
            internal timestep.
    """

    def __init__(self,
                 x,
                 sigma_min,
                 sigma_max,
                 seed=None,
                 transform=lambda x: x):
        self.transform = transform
        t0 = self.transform(torch.as_tensor(sigma_min))
        t1 = self.transform(torch.as_tensor(sigma_max))
        self.tree = BatchedBrownianTree(x, t0, t1, seed)

    def __call__(self, sigma, sigma_next):
        t0 = self.transform(torch.as_tensor(sigma))
        t1 = self.transform(torch.as_tensor(sigma_next))
        return self.tree(t0, t1) / (t1 - t0).abs().sqrt()


@torch.no_grad()
def sample_dpmpp_2m_sde(noise,
                        model,
                        sigmas,
                        eta=1.,
                        s_noise=1.,
                        solver_type='midpoint',
                        show_progress=True,
                        variant_info=None,
                        progress_callback=None,
                        seed=None):
    """
    DPM-Solver++ (2M) SDE.
    """
    assert solver_type in {'heun', 'midpoint'}

    x = noise * sigmas[0]
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas[
        sigmas < float('inf')].max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed if seed >= 0 else None)
    old_denoised = None
    h_last = None

    total_steps_in_solver = len(sigmas) - 1
    for i in trange(total_steps_in_solver, disable=not show_progress, desc="Diffusion Steps"):
        # Check for cancellation at the start of each diffusion step
        try:
            from logic.cancellation_manager import cancellation_manager
            cancellation_manager.check_cancel()
        except ImportError:
            # If cancellation manager is not available, continue without cancellation
            pass
        except Exception as e:
            # If there's a CancelledError or other cancellation exception, re-raise it
            if "cancel" in str(e).lower() or "cancelled" in str(e).lower():
                raise e
            # For other exceptions, continue without cancellation
            pass
        
        current_step = i + 1
        # logger.info(f'Diffusion Step: {current_step}/{total_steps_in_solver}')
        if progress_callback:
            progress_callback(step=current_step, total_steps=total_steps_in_solver)

        if sigmas[i] == float('inf'):
            # Euler method
            denoised = model(noise, sigmas[i], variant_info=variant_info)
            x = denoised + sigmas[i + 1] * noise
        else:
            _, c_in = get_scalings(sigmas[i])
            denoised = model(x * c_in, sigmas[i], variant_info=variant_info)
            if sigmas[i + 1] == 0:
                # Denoising step
                x = denoised
            else:
                # DPM-Solver++(2M) SDE
                t, s = -sigmas[i].log(), -sigmas[i + 1].log()
                h = s - t
                eta_h = eta * h

                x = sigmas[i + 1] / sigmas[i] * (-eta_h).exp() * x + \
                    (-h - eta_h).expm1().neg() * denoised

                if old_denoised is not None:
                    r = h_last / h
                    if solver_type == 'heun':
                        x = x + ((-h - eta_h).expm1().neg() / (-h - eta_h) + 1) * \
                            (1 / r) * (denoised - old_denoised)
                    elif solver_type == 'midpoint':
                        x = x + 0.5 * (-h - eta_h).expm1().neg() * \
                            (1 / r) * (denoised - old_denoised)

                x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * sigmas[
                    i + 1] * (-2 * eta_h).expm1().neg().sqrt() * s_noise

            old_denoised = denoised
            h_last = h

    if variant_info is not None and variant_info.get('type') == 'variant1':
        x_long, x_short = x.chunk(2, dim=0)
        x = x_long * (1-variant_info['alpha']) + x_short * variant_info['alpha']

    return x
import torch as torch
import numpy as np


def sample_image(
    input_image: torch.Tensor,  # the initial photon image (batch, channel, _ | z, x, y)
    model: torch.nn.Module,  # the network used to predict the next phton location
    dim: int = 3,  # dimension of the input image / model (new)
    max_photons: int = None,  # stop sampling when image contains more photons
    max_its: int = 500000,  # stop sampling after max_its iterations
    max_psnr: int = -15,  # stop sampling when pseudo PSNR is larger max_psnr
    save_every_n: int = 5,  # store and return images at every nth step
    beta: float = 0.1,  # photon number is increased exponentially by factor beta in each step.
) -> tuple:
    """
    Samples an image using Generative Accumulation of Photons (GAP) based on an initial photon image.
    If the initial photon image contains only zeros the model samples from scratch.
    If it contains photon numbers, the model performs diversity denoising.
    """
    assert dim in [2, 3], "dim must be 2 or 3"
    
    photons = input_image.clone() 
    denoised = None
    
    stack = []
    for i in range(max_its):
        # compute the pseudo PSNR
        psnr = np.log10(photons.mean().item() + 1e-50) * 10
        psnr = max(-40, psnr)
        # stop if we have enough photons or the PSNR is large enough
        if (max_photons is not None) and (photons.sum().item() > max_photons):
            break
        if psnr > max_psnr:
            break
        # predict the next photon locations
        denoised = model(photons).detach()
        denoised = denoised - denoised.max()
        denoised = torch.exp(denoised)
        if dim == 2:
            denoised = denoised / (denoised.sum(dim=(-1, -2, -3), keepdim=True))
        else:
            denoised = denoised / (denoised.sum(dim=(-1, -2, -3, -4), keepdim=True))

        # here we save an image into our stack
        if (save_every_n is not None) and (i % save_every_n == 0):
            imgsave = denoised[0, 0, :, ...].detach().cpu()
            imgsave = imgsave / imgsave.max()
            photsave = photons[0, 0, :, ...].detach().cpu()
            photsave = photsave / max(photsave.max(), 1)
            combi = torch.cat((photsave, imgsave), -1)
            stack.append(combi.numpy())

        # increase photon number
        photnum = max(beta * photons.sum(), 1)

        # draw new photons
        new_photons = torch.poisson(denoised * photnum)

        # add new photons
        photons += new_photons

    return (
        denoised[...].detach().cpu().numpy(),
        photons[...].detach().cpu().numpy(),
        stack,
        i,
    )

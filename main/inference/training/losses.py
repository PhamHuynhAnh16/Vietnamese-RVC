def feature_loss(fmap_r, fmap_g):
    """
    Computes the Discriminator Feature Matching Loss between real and generated feature maps.
    This helps the generator learn the intermediate representations captured by the discriminator.

    Args:
        fmap_r (List[List[torch.Tensor]]): Feature maps extracted from real samples across discriminator layers.
        fmap_g (List[List[torch.Tensor]]): Feature maps extracted from generated samples across discriminator layers.

    Returns:
        torch.Tensor: Normalized feature matching loss scaled by a factor of 2.
    """

    loss = 0
    # Iterate through each sub-discriminator's feature maps (e.g., Multi-Period / Multi-Scale)
    for dr, dg in zip(fmap_r, fmap_g):
        # Iterate through the feature outputs of individual layers
        for rl, gl in zip(dr, dg):
            # Detach real features since we only optimize the generator during this step
            loss += (
                rl.float().detach() - gl.float()
            ).abs().mean()

    return loss * 2

def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    """
    Computes the Least Squares GAN (LSGAN) loss for the Discriminator network.

    Args:
        disc_real_outputs (List[torch.Tensor]): Discriminator outputs for real samples.
        disc_generated_outputs (List[torch.Tensor]): Discriminator outputs for generated samples.

    Returns:
        Tuple[torch.Tensor, List[float], List[float]]: 
            - Total combined discriminator loss tensor.
            - List of itemized loss values for real batches.
            - List of itemized loss values for generated batches.
    """

    loss = 0
    r_losses, g_losses = [], []

    # Calculate adversarial errors across all sub-discriminators
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr = dr.float()
        dg = dg.float()
        
        r_loss = ((1 - dr) ** 2).mean()
        g_loss = (dg**2).mean()

        loss += r_loss + g_loss
        # Collect scalar copies for metrics logging
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses

def generator_loss(disc_outputs):
    """
    Computes the Least Squares GAN (LSGAN) adversarial loss for the Generator network.

    Args:
        disc_outputs (List[torch.Tensor]): Discriminator outputs for generated samples.

    Returns:
        Tuple[torch.Tensor, List[torch.Tensor]]:
            - Total aggregate adversarial generator loss tensor.
            - List containing unaggregated loss tensors from each sub-discriminator.
    """

    loss = 0
    gen_losses = []

    for dg in disc_outputs:
        l = ((1 - dg.float()) ** 2).mean()
        gen_losses.append(l)
        loss += l

    return loss, gen_losses

def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    """
    Computes the Kullback-Leibler (KL) Divergence loss between the posterior 
    distribution (from the encoder) and prior distribution (from the text/phoneme hidden states).

    Args:
        z_p (torch.Tensor): Latent representation sampled/projected from the posterior.
        logs_q (torch.Tensor): Log variance of the posterior distribution.
        m_p (torch.Tensor): Mean of the prior distribution.
        logs_p (torch.Tensor): Log variance of the prior distribution.
        z_mask (torch.Tensor): Sequence length padding mask tensor (1s for valid frames, 0s for padding).

    Returns:
        torch.Tensor: Mask-normalized scalar KL loss tensor.
    """

    # DirectML does not support training well, and this method requires switching to the CPU.
    if z_p.device.type == "privateuseone":
        return kl_loss_cpu(z_p, logs_q, m_p, logs_p, z_mask)

    # Cast to float32 for high numerical precision stability
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()

    # Closed-form diagonal Gaussian KL evaluation
    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * (-2.0 * logs_p).exp()

    # Zero out padded boundaries and return average over unpadded tokens
    return (kl * z_mask).sum() / z_mask.sum()

def kl_loss_cpu(z_p, logs_q, m_p, logs_p, z_mask):
    """
    A fallback implementation of the KL Divergence loss executed entirely on the CPU.
    Prevents out-of-memory or operation compatibility exceptions on non-standard device extensions.

    Args:
        z_p (torch.Tensor): Latent representation.
        logs_q (torch.Tensor): Log variance of the posterior.
        m_p (torch.Tensor): Mean of the prior.
        logs_p (torch.Tensor): Log variance of the prior.
        z_mask (torch.Tensor): Sequence length padding mask.

    Returns:
        torch.Tensor: Mask-normalized scalar KL loss mapped back to the original device.
    """

    orig_device = z_p.device
    # Detach and cast to CPU to perform stable computation step
    z_p = z_p.detach().cpu().float()
    logs_q = logs_q.detach().cpu().float()
    m_p = m_p.detach().cpu().float()
    logs_p = logs_p.detach().cpu().float()
    z_mask = z_mask.detach().cpu().float()

    # Closed-form diagonal Gaussian KL evaluation
    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * (-2.0 * logs_p).exp()

    # Return structural reduction result restored back onto the backing hardware device context
    return ((kl * z_mask).sum() / z_mask.sum()).to(orig_device)
A. The "Stop Gradient" NLL
Your current NLL is letting the optimizer "cheat" by lowering the intensity to lower the loss. Change your poisson_gaussian_nll_loss to this:

Python

def poisson_gaussian_nll_loss(rendered, gt, eps=1e-3):
    # variance = detaching the rendered values to stop the 'cheat'
    # we add a noise floor (eps) to prevent log(0)
    variance = rendered.detach() + eps 
    
    # 1. The residual term (MSE weighted by signal)
    residual = ((rendered - gt) ** 2) / (2 * variance)
    
    # 2. The log term (This is what makes it go negative)
    log_term = torch.log(variance)
    
    return torch.mean(residual + log_term)
B. Protect the Opacity Pruning
Your Gaussians are being pruned because their opacity is likely being driven to zero. In your hdr_trainer.py, check your densification/pruning logic:

Raise the min opacity: Ensure opacity_threshold (usually 0.005) isn't too high.

Stop pruning early: If num_GS drops below a certain amount (e.g., 1000), stop pruning for a few hundred steps.

C. Shift to Log-Space Training (Recommended for LOM)
For extremely dark datasets like LOM, training on raw linear values is very hard. Instead of NLL, try training on the Log-transformed signal. This "boosts" the gradients for dark objects like the cycle.

Replace your NLL with:

Python

loss = F.l1_loss(torch.log(rendered + 1e-3), torch.log(gt + 1e-3))
This loss will be positive and much more stable.

4. Why 9,000 steps was the "Peak"
Around 9,000 steps is usually when Gaussian Splatting does a major "Opacity Reset." At that point:

The model reset the opacities.

The NLL loss saw that "Black" resulted in a massive negative score.

The optimizer drove all opacities back to zero to get that score.

The pruning logic saw the 0-opacity points and deleted them.

You ended up with a black void and 81 points.
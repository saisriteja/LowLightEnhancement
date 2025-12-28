I need to debug the existing code, now I need to save the intermeidate outputs at every 5K iteration(variable) so that i can debug each module whats happening


Test these sequentially:
a) Disable all MLPs temporarily - render without perturbations

If this looks good → MLP is destroying geometry
If this looks bad → base Gaussian training is broken

b) Render with only illumination MLP (disable perturbation MLP)

Should only change brightness, not structure
If structure changes → illumination MLP is too powerful
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
from scipy.optimize import minimize_scalar

def run_ddm_animation():
    # Set device (ROCm uses 'cuda' interface for AMD GPUs)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Fully vectorized Wiener first passage time density (batched over t and k)
    def wfpt_density(t, v, a, z, err=1e-15):
        if torch.all(t <= 0):
            return torch.zeros_like(t)
        
        # Scalars to tensors on device
        v = torch.tensor(v, device=device, dtype=torch.float64)
        a = torch.tensor(a, device=device, dtype=torch.float64)
        z = torch.tensor(z, device=device, dtype=torch.float64)
        err = torch.tensor(err, device=device, dtype=torch.float64)
        
        w = z / a
        tt = t / (a ** 2)
        pi = torch.pi
        sqrt = torch.sqrt
        log = torch.log
        exp = torch.exp
        sin = torch.sin
        
        # Vectorized kl for large-time
        kl_mask_small = (pi * tt * err < 1)
        kl = torch.zeros_like(tt)
        kl[~kl_mask_small] = 1 / (pi * sqrt(tt[~kl_mask_small]))
        kl[kl_mask_small] = torch.clamp(sqrt(-2 * log(pi * tt[kl_mask_small] * err) / (pi**2 * tt[kl_mask_small])), 
                                        min=1 / (pi * sqrt(tt[kl_mask_small])))
        
        # Vectorized ks for small-time
        ks = torch.full_like(tt, 2.0)
        ks_mask_small = (2 * sqrt(2 * pi * tt) * err < 1)
        ks[ks_mask_small] = torch.clamp(2 + sqrt(-2 * tt[ks_mask_small] * log(2 * sqrt(2 * pi * tt[ks_mask_small]) * err)), 
                                        min=sqrt(tt[ks_mask_small]) + 1)
        
        # Decide expansion per t
        small_mask = (ks < kl)
        p = torch.zeros_like(tt)
        
        # Small-time expansion: Broadcast k over t using meshgrid-like
        if torch.any(small_mask):
            tt_small = tt[small_mask]
            n_small = len(tt_small)
            max_k_off = 5  # Reasonable bound for convergence, symmetric
            k_offs = torch.arange(-max_k_off, max_k_off + 1, device=device, dtype=torch.float64).unsqueeze(1).expand(-1, n_small)
            tt_expand = tt_small.unsqueeze(0).expand(len(k_offs), -1)
            w_expand = w.unsqueeze(0).expand_as(k_offs)
            
            terms = (w_expand + 2 * k_offs) * exp(- (w_expand + 2 * k_offs)**2 / (2 * tt_expand))
            p_small = terms.sum(dim=0) / sqrt(2 * pi * tt_small**3)
            p[small_mask] = p_small
        
        # Large-time expansion: Batched k=1 to max_K
        if torch.any(~small_mask):
            tt_large = tt[~small_mask]
            n_large = len(tt_large)
            kl_large = kl[~small_mask]
            max_k_large = int(kl_large.max().item()) + 1
            k = torch.arange(1, max_k_large + 1, device=device, dtype=torch.float64).unsqueeze(1).expand(-1, n_large)
            tt_expand = tt_large.unsqueeze(0).expand(len(k), -1)
            w_expand = w.unsqueeze(0).expand_as(k)
            
            exp_terms = exp(- k**2 * pi**2 * tt_expand / 2)
            sin_terms = sin(k * pi * w_expand)
            p_large = (k * exp_terms * sin_terms).sum(dim=0) * pi
            p[~small_mask] = p_large
        
        # Scale
        scale = exp(-v * a * w - (v**2 * t) / 2) / (a**2)
        p *= scale
        p = torch.clamp(p, min=1e-300)
        
        return p

    # Set up the figure and axes
    fig, axs = plt.subplots(2, 1, figsize=(8, 6))

    # Process axis
    ax = axs[0]
    ax.set_xlim(0, 5)
    ax.set_ylim(-0.1, 1.1)
    ax.set_title('Real-time Drift Diffusion Model Simulation')
    ax.set_xlabel('Time')
    ax.set_ylabel('Evidence')
    ax.axhline(0, color='red', linestyle='--', label='Lower Boundary')
    ax.axhline(1, color='green', linestyle='--', label='Upper Boundary')
    ax.legend()

    # Initialize empty line for process
    line, = ax.plot([], [], lw=2, color='blue')

    # Parameter axis
    param_ax = axs[1]
    param_ax.set_xlim(0, 1000)
    param_ax.set_ylim(0, 1)
    param_ax.set_title('Estimated Drift Parameter')
    param_ax.set_xlabel('Number of Trials')
    param_ax.set_ylabel('Estimated v')
    param_ax.axhline(0.5, color='gray', linestyle='--', label='True v=0.5')
    param_ax.legend()

    # Initialize line for estimated v
    v_line, = param_ax.plot([], [], color='red', label='Estimated v')

    # Parameters as tensors
    dt = torch.tensor(0.001, device=device)
    a = torch.tensor(1.0, device=device)
    z = torch.tensor(0.5, device=device)
    v_true = torch.tensor(0.5, device=device)
    sigma = torch.tensor(1.0, device=device)

    # Global tensors on device
    t = torch.tensor(0.0, device=device)
    x = z.clone()
    times = torch.zeros(1, device=device)
    values = torch.zeros(1, device=device)
    times[0] = 0.0
    values[0] = z.item()

    rts = torch.empty(0, device=device, dtype=torch.float64)
    choices = torch.empty(0, device=device, dtype=torch.long)
    trial_number = 0
    estimate_times = []
    estimated_vs = []
    steps_per_frame = 10

    # Initialization function
    def init():
        line.set_data([], [])
        v_line.set_data([], [])
        return line, v_line

    # Animation update function
    def animate(frame):
        global t, x, times, values, rts, choices, trial_number
        hit_boundary = False
        
        for _ in range(steps_per_frame):
            t += dt
            # GPU-accelerated increment
            dx_drift = v_true * dt
            dx_diff = sigma * torch.sqrt(dt) * torch.randn(1, device=device)
            dx = dx_drift + dx_diff
            x += dx
            # Append to path tensors
            times = torch.cat([times, t.unsqueeze(0)])
            values = torch.cat([values, x.unsqueeze(0)])
            
            if x >= a or x <= 0:
                choice = torch.tensor(1 if x >= a else 0, device=device, dtype=torch.long)
                rt = t.clone()
                rts = torch.cat([rts, rt.unsqueeze(0)])
                choices = torch.cat([choices, choice.unsqueeze(0)])
                trial_number += 1
                # Reset tensors for next trial
                t = torch.tensor(0.0, device=device)
                x = z.clone()
                times = torch.zeros(1, device=device)
                values = torch.zeros(1, device=device)
                times[0] = 0.0
                values[0] = z.item()
                hit_boundary = True
                break
        
        # Update process line (to numpy)
        line.set_data(times.cpu().numpy(), values.cpu().numpy())
        
        # Adjust x-limits
        if t.item() > ax.get_xlim()[1]:
            ax.set_xlim(0, t.item() + 1)
        
        # Fit every 50 trials
        if trial_number % 50 == 0 and trial_number > 0:
            def neg_loglik(v_val):
                v_tensor = torch.tensor(v_val, device=device, dtype=torch.float64)
                upper_mask = (choices == 1)
                lower_mask = (choices == 0)
                p_upper = wfpt_density(rts[upper_mask], v_tensor, a.item(), z.item())
                p_lower = wfpt_density(rts[lower_mask], -v_tensor, a.item(), z.item())
                ll = torch.sum(torch.log(p_upper)) + torch.sum(torch.log(p_lower))
                return -ll.item()
            
            res = minimize_scalar(neg_loglik, bounds=(0, 2), method='bounded')
            est_v = res.x if res.success else np.nan
            estimated_vs.append(est_v)
            estimate_times.append(trial_number)
            
            # Update parameter line
            v_line.set_data(estimate_times, estimated_vs)
            
            # Adjust param x-limits
            if trial_number > param_ax.get_xlim()[1]:
                param_ax.set_xlim(0, trial_number + 100)
            
            param_ax.figure.canvas.draw()
        
        ax.figure.canvas.draw()
        return line, v_line

    # Create the animation
    ani = FuncAnimation(fig, animate, init_func=init, frames=100000, interval=1, blit=True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_ddm_animation()

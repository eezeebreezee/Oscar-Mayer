import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch

def run_wiener_animation():
    # Set device (ROCm uses 'cuda' interface for AMD GPUs)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set up the figure and axis
    fig, ax = plt.subplots()
    ax.set_xlim(0, 10)  # Time horizon from 0 to 10 units
    ax.set_ylim(-10, 10)  # Adjust y-limits based on expected volatility
    ax.set_title('Real-time Wiener Process (Brownian Motion)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')

    # Initialize empty line
    line, = ax.plot([], [], lw=2, color='blue')

    # Parameters as tensors on device
    dt = torch.tensor(0.01, device=device)
    t = torch.tensor(0.0, device=device)
    w = torch.tensor(0.0, device=device)

    # Tensors to store the path (on device for GPU accumulation)
    times = torch.zeros(1, device=device)
    values = torch.zeros(1, device=device)
    times[0] = 0.0
    values[0] = 0.0

    # Initialization function
    def init():
        line.set_data([], [])
        return line,

    # Animation update function
    def animate(frame):
        global t, w, times, values
        t += dt
        # GPU-accelerated random increment: sqrt(dt) * N(0,1)
        dw = torch.sqrt(dt) * torch.randn(1, device=device)
        w += dw
        # Append to tensors (resize and copy)
        times = torch.cat([times, t.unsqueeze(0)])
        values = torch.cat([values, w.unsqueeze(0)])
        
        # Update the line data (transfer to CPU numpy)
        line.set_data(times.cpu().numpy(), values.cpu().numpy())
        
        # Dynamically adjust y-limits if needed (use CPU min/max for simplicity)
        val_np = values.cpu().numpy()
        if w.item() > ax.get_ylim()[1] or w.item() < ax.get_ylim()[0]:
            ax.set_ylim(np.min(val_np) - 1, np.max(val_np) + 1)
            ax.figure.canvas.draw()
        return line,

    # Create the animation
    ani = FuncAnimation(fig, animate, init_func=init, frames=10000, interval=10, blit=True)
    plt.show()

if __name__ == "__main__":
    run_wiener_animation()

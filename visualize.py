import matplotlib.pyplot as plt
import numpy as np
import os
# ---------------------------
# Visualization for debugging
# ---------------------------
def plot_traces_heatmap(traces_normalized, max_var, min_corr):
    """Plot all normalized traces as a heatmap."""
    title = f"Normalized Traces, min_var: {max_var}, min_corr: {min_corr}"
    fig, ax = plt.subplots(figsize=(14, max(6, traces_normalized.shape[0] * 0.15)))
    
    im = ax.imshow(traces_normalized, aspect='auto', cmap='magma', 
                   interpolation='nearest', vmin=0, vmax=1)
    
    ax.set_xlabel('Time (frames)', fontsize=12)
    ax.set_ylabel('Neuron ID', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Normalized Activity', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(f'all_neuron_traces_heatmap.png', dpi=150, bbox_inches='tight')
    print(f"  Saved heatmap to 'all_neuron_traces_heatmap.png'")
    plt.close()



def plot_cont_mask(corr_img, title = "Correlation Map", save_path="media/corr_map.png"):
    """
    Display a 2D correlation image (corr_mask) as a heatmap with a colorbar,
    and optionally save it to a file.

    Parameters
    ----------
    corr_img : 2D numpy array (H x W)
        The correlation strength for each pixel.
    title : str
        Title for the plot.
    save_path : str or None
        If provided, the plot will be saved to this filepath (e.g. 'corr_map.png').
    """
    plt.figure(figsize=(6, 6))
    im = plt.imshow(corr_img, cmap="viridis", interpolation="nearest")
    plt.colorbar(im, label="Correlation")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()

    if save_path is not None:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close()


def plot_bin_mask(mask, title, filepath="media/bin_mask.png"):
    """
    Plot a binary ROI mask (boolean array). True pixels are shown in white.

    Parameters
    ----------
    mask : 2D numpy array (H x W), boolean
        Binary mask where True indicates ROI pixels.
    title : str
        Plot title.
    save_path : str or None
        If provided, saves the plot to this path.
    """

    plt.figure(figsize=(5,5))
    plt.imshow(mask.astype(float), cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()

    if filepath is not None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        plt.savefig(filepath, dpi=300, bbox_inches="tight")

    plt.close()

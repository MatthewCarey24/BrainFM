from scipy import io
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import remove_small_objects    
from scipy.ndimage import label
from visualize import *
import config
from scipy.ndimage import gaussian_filter1d



# ---------------------------
# Load data
# ---------------------------
def load_voltage_imaging_data(filepath):
    data = io.loadmat(filepath)
    movie = data['movie']
    metadata = data['metadata']
    stimulation_signal = data['stimulationSignal'].flatten()

    # convert to float32 to prevent negatives being huge (2s compliment)
    # movie = movie.astype(np.float32) - metadata['darkLevel']  
    
    print(f"Movie shape: {movie.shape}")
    print(f"Movie dimensions: {movie.shape[0]}x{movie.shape[1]} pixels, {movie.shape[2]} timepoints")
    
    return movie, metadata, stimulation_signal


def compute_corr_mask(movie):
    ############################################################################
    # compute a mask where each position is how correlated that pixel is to it's 
    # neighbors, inspired by Suite2P's math
    ############################################################################

    height, width, frames = movie.shape
    corr_mask = np.zeros((height, width))

    # compute variance mask and std mask over timefor normalizion
    variance_mask = np.var(movie, axis=2)
    std_mask = np.sqrt(variance_mask + 1e-10)

    # center traces around their mean for correlation formula
    mean_mask = np.mean(movie, axis=2, keepdims=True)
    centered_movie = movie - mean_mask

    # by offset is faster than looking up, though our dimensions arent huge
    neighbor_offsets = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1)
    ]
    for dx, dy in neighbor_offsets:
        # movie where each pos (i,j) is the trace of (i + dx, j + dy)
        shifted_movie = np.roll(np.roll(centered_movie, dx, axis=0), dy, axis=1)

        # (160, 800, 1478), where product[x,y] is the element wise product if 
        # pixel (i,j)s trace with its neighbor defined by dx, dy (wrapping?)
        product_movie = centered_movie * shifted_movie
        
        # (160, 800), average across time to get a mask
        covariance_mask = np.mean(product_movie, axis=2)

        shifted_std_mask = np.roll(np.roll(std_mask, dy, axis=0), dx, axis=1)

        neighbor_corr_mask = covariance_mask / (std_mask * shifted_std_mask + 1e-10)
        corr_mask += neighbor_corr_mask
    
    corr_mask /= len(neighbor_offsets)

    return corr_mask



def compute_spike_mask(movie):
    # Temporal derivative emphasizes spikes
    diff_movie = np.diff(movie, axis=2)
    spike_mask = np.var(diff_movie, axis=2)

    return spike_mask


def extract_roi_masks(movie):
    min_corr = config.MIN_CORRELATION_PERCENTILE
    min_size = config.MIN_REGION_SIZE
    max_size = config.MAX_REGION_SIZE

    neuron_masks = []

    corr_mask = compute_corr_mask(movie)
    plot_cont_mask(corr_mask, save_path="media/corr_mask.png")
    bin_corr_mask = corr_mask > np.percentile(corr_mask, config.MIN_CORRELATION_PERCENTILE)
    plot_bin_mask(bin_corr_mask, title=f"Bin Corr Mask, min_corr: {min_corr}", filepath="media/bin_corr_mask")

    spike_mask = compute_spike_mask(movie)
    plot_cont_mask(spike_mask, title="Spike Var Heatmap", save_path="media/spike_var_heatmap.png")
    bin_spike_mask = spike_mask < np.percentile(spike_mask, config.MAX_DER_VAR)
    plot_bin_mask(bin_spike_mask, title="Bin Spike Mask", filepath="media/bin_spike_mask")

    corr_var_mask = bin_spike_mask & bin_corr_mask
    corr_var_mask = remove_small_objects(corr_var_mask, min_size)
    plot_bin_mask(corr_var_mask, title="corr and spike mask", filepath=f"media/corr_and_spike.png")

    labeled_mask, num_regions = label(corr_var_mask)
    print(f"Found and labeled {num_regions} distinct regions.")

    for region_id in range(1, num_regions + 1):
        # extract bin mask where the labeled mask is i for ith neuron
        neuron_mask = (labeled_mask == region_id) 
        if np.sum(neuron_mask) <= max_size:
            neuron_masks.append(neuron_mask)
        
    print(f"Kept {len(neuron_masks)} masks")
    return neuron_masks




def flatten_roi_trace(roi_mask, movie):
    # indexing flattens spatial dimensions, then average them
    return movie[roi_mask].mean(axis=0)



def apply_dff_norm(trace):
    f0 = np.mean(trace[1320:])
    # multiply numerator to emphasize spikes
    norm_trace = (trace - f0) * config.SPIKE_MULT / f0
    return norm_trace



if __name__ == "__main__":
    filepath = "movie.mat"
    movie, metadata, stimulation = load_voltage_imaging_data(filepath)

    normalized_traces = []

    neuron_masks = extract_roi_masks(movie)

    for neuron_mask in neuron_masks:
        trace = flatten_roi_trace(neuron_mask, movie)
        normalized_trace = apply_dff_norm(trace)

        normalized_traces.append(normalized_trace)
    
    normalized_traces = np.array(normalized_traces)
    plot_traces_heatmap(normalized_traces, max_var=config.MAX_DER_VAR, min_corr=config.MIN_CORRELATION_PERCENTILE)
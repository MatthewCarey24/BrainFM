import numpy as np
import matplotlib.pyplot as plt
from scipy import io
import os
from PIL import Image
import imageio
import matplotlib.cm as cm


# ---------------------------
# Load data
# ---------------------------
def load_voltage_imaging_data(filepath):
    data = io.loadmat(filepath)

    movie = data['movie']
    metadata = data['metadata']
    stimulation_signal = data['stimulationSignal'].flatten()

    print(f"Movie shape: {movie.shape}")
    print(f"Movie dimensions: {movie.shape[0]}x{movie.shape[1]} pixels, {movie.shape[2]} timepoints")
    print(f"Stimulation signal shape: {stimulation_signal.shape}")

    return movie, metadata, stimulation_signal


# ---------------------------
# Image processing helpers
# ---------------------------
def apply_gamma(img, gamma=0.7):
    img_float = img.astype(np.float32) / 255.0
    img_gamma = np.power(img_float, gamma)
    return np.clip(img_gamma * 255, 0, 255).astype(np.uint8)


def gentle_per_frame_stretch(frame_uint8, low=2, high=98):
    """Simple percentile-based stretch."""
    p_low = np.percentile(frame_uint8, low)
    p_high = np.percentile(frame_uint8, high)

    stretched = (frame_uint8 - frame_uint8.min()) / (frame_uint8.max() - frame_uint8.min() + 1e-8)
    stretched = np.clip(stretched * 255, 0, 255).astype(np.uint8)

    return stretched


def apply_colormap(frame_uint8, cmap_name="magma"):
    """Apply a Matplotlib colormap to 8-bit image."""
    cmap = cm.get_cmap(cmap_name)
    colored = cmap(frame_uint8 / 255.0)[:, :, :3]  # Drop alpha
    return (colored * 255).astype(np.uint8)


# ---------------------------
# Main movie creation with flash enhancement
# ---------------------------
def create_activity_movie(movie, stimulation_signal, metadata,
                          output_filename='movie.mp4', start_frame=0,
                          flash_emphasis=2.0):

    end_frame = 1320  # or movie.shape[2]
    frames_to_save = []

    # Metadata
    sample_time = metadata['sampleTime'].item().item()
    dark_level = metadata['darkLevel'].item().item()
    real_fps = metadata['sampleRate'].item().item()
    playback_fps = real_fps / 10

    # Subtract dark level
    movie_corrected = movie.astype(float) - dark_level

    # ---------------------------
    # Calculate baseline for flash detection
    # ---------------------------
    print("\nCalculating baseline for flash enhancement...")
    baseline = np.percentile(movie_corrected, 20, axis=2, keepdims=True)
    
    # --- Global normalization range (for baseline) ---
    global_min = np.percentile(baseline, 1)
    global_max = np.percentile(movie_corrected, 99)

    print(f"\nBaseline range: {global_min:.2f} to {np.percentile(baseline, 99):.2f}")
    print(f"Peak range: {np.percentile(movie_corrected, 95):.2f} to {global_max:.2f}")
    print(f"Flash emphasis factor: {flash_emphasis}x")
    print(f"\nPreparing {end_frame - start_frame} frames...")

    # ---------------------------
    # Frame loop
    # ---------------------------
    for i in range(start_frame, end_frame):

        # Extract raw frame and its baseline
        frame = movie_corrected[:, :, i]
        frame_baseline = baseline[:, :, 0]
        
        # Calculate deviation from baseline
        deviation = frame - frame_baseline
        
        # Emphasize positive deviations (flashes) more than baseline
        # Keep baseline as-is, but boost activity above baseline
        frame_enhanced = frame_baseline + deviation * flash_emphasis
        
        # Clip to prevent overshooting
        frame_enhanced = np.clip(frame_enhanced, global_min, global_max * 1.2)

        # === Stage 1: Global normalization ===
        frame_norm = (frame_enhanced - global_min) / (global_max - global_min + 1e-8)
        frame_norm = np.clip(frame_norm * 255, 0, 255).astype(np.uint8)

        # === Gamma correction ===
        frame_norm = apply_gamma(frame_norm, gamma=1.2)  # Slightly lower gamma to show more detail

        # === Stage 2: Gentle local contrast ===
        frame_final = gentle_per_frame_stretch(frame_norm, low=1, high=99)

        # === Apply colormap ===
        colored = apply_colormap(frame_final, cmap_name="magma")

        frames_to_save.append(colored)

        if (i - start_frame) % 100 == 0:
            print(f"  Processed {i - start_frame} frames...")

    # ---------------------------
    # Save the movie
    # ---------------------------
    print(f"\nSaving video to {output_filename} at {playback_fps:.1f} FPS...")
    imageio.mimsave(output_filename, frames_to_save, fps=playback_fps)

    print(f"Done! {len(frames_to_save)} frames saved.")
    print(f"File size: {os.path.getsize(output_filename) / 1024 / 1024:.1f} MB")

    return output_filename


# ---------------------------
# Run script
# ---------------------------
if __name__ == "__main__":
    filepath = "movie.mat"

    movie, metadata, stim = load_voltage_imaging_data(filepath)

    create_activity_movie(
        movie, stim, metadata,
        output_filename='movie.mp4',
        flash_emphasis=4.0  # Increase to 3.0 or 4.0 for even stronger emphasis
    )
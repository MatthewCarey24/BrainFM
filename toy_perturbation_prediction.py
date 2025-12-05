import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

class NeuralPerturbationTransformer(nn.Module):
    """
    Transformer model for predicting perturbations from neural activity.
    
    Args:
        n_neurons: Number of neurons recorded
        patch_size: Duration of each patch token (in time steps)
        d_model: Latent dimension size
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        n_perturbations: Number of perturbation classes
        dropout: Dropout rate
        use_masking: Whether to use masked pretraining
    """
    def __init__(
        self,
        n_neurons,
        patch_size,
        d_model=256,
        n_heads=8,
        n_layers=6,
        n_perturbations=34000, # 24,000 gene knockouts + 10,000 drugs
        dropout=0.1,
        use_masking=False
    ):
        super().__init__()
        
        self.n_neurons = n_neurons
        self.patch_size = patch_size
        self.d_model = d_model
        self.use_masking = use_masking
        
        # Patch embedding: linear projection from N neurons to d_model dimensions
        self.patch_embedding = nn.Linear(n_neurons, d_model)
        
        # learnable CLS token, inference point for perturbation prediction
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len=5000)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Classification head (maps CLS token to perturbation logits)
        self.classifier = nn.Linear(d_model, n_perturbations)
        
        # For masked pretraining: reconstruction head
        if use_masking:
            self.reconstruction_head = nn.Linear(d_model, n_neurons)
            self.mask_token = nn.Parameter(torch.randn(1, 1, d_model))
    
    def patchify(self, neural_data):
        """
        Convert neural data into patch tokens

        Could transpose this, ie each token is a neuron and vector is values 
        over time. removes need for spatial encoding, makes more sense to learn 
        interrelationship between neurons but temporal data ive mostlty seen with 
        tokens as timesteps
        
        Args:
            neural_data: (batch, time, n_neurons)
        Returns:
            patches: (batch, n_patches, n_neurons)
        """
        batch_size, time, n_neurons = neural_data.shape
        
        # Reshape into patches
        n_patches = time // self.patch_size
        patches = neural_data[:, :n_patches * self.patch_size, :]
        patches = patches.reshape(batch_size, n_patches, self.patch_size, n_neurons)
        
        # Average over patch duration (could also use other aggregation)
        patches = patches.mean(dim=2)  # (batch, n_patches, n_neurons)
        
        return patches
    
    def forward(self, neural_data, mask_ratio=0.0):
        """
        Forward pass.
        
        Args:
            neural_data: (batch, time, n_neurons)
            mask_ratio: Proportion of patches to mask (for pretraining)
        Returns:
            logits: (batch, n_perturbations) - perturbation predictions
            reconstruction: (batch, n_patches, n_neurons) - reconstructed patches (if masking)
        """
        batch_size = neural_data.shape[0]
        
        # Patchify the neural data
        patches = self.patchify(neural_data)  # (batch, n_patches, n_neurons)
        n_patches = patches.shape[1]
        
        # Apply masking if needed (for pretraining)
        masked_patches = patches
        mask_indices = None
        if mask_ratio > 0 and self.use_masking:
            masked_patches, mask_indices = self.apply_masking(patches, mask_ratio)
        
        # Embed patches
        embeddings = self.patch_embedding(masked_patches)  # (batch, n_patches, d_model)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat([cls_tokens, embeddings], dim=1)  # (batch, n_patches+1, d_model)
        
        # Add positional encoding
        embeddings = self.pos_encoding(embeddings)
        
        # Transformer encoding
        encoded = self.transformer(embeddings)  # (batch, n_patches+1, d_model)
        
        # Extract CLS token representation
        cls_output = encoded[:, 0, :]  # (batch, d_model)
        
        # Classification
        logits = self.classifier(cls_output)  # (batch, n_perturbations)
        
        # Reconstruction (for masked pretraining)
        reconstruction = None
        if self.use_masking and mask_indices is not None:
            patch_outputs = encoded[:, 1:, :]  # (batch, n_patches, d_model)
            reconstruction = self.reconstruction_head(patch_outputs)  # (batch, n_patches, n_neurons)
        
        return logits, reconstruction, mask_indices
    
    def apply_masking(self, patches, mask_ratio):
        """
        Randomly mask patches for self-supervised pretraining.
        
        Args:
            patches: (batch, n_patches, n_neurons)
            mask_ratio: Proportion of patches to mask
        Returns:
            masked_patches: patches with some replaced by mask token
            mask_indices: boolean mask of which patches were masked
        """
        batch_size, n_patches, _ = patches.shape
        n_masked = int(n_patches * mask_ratio)
        
        # Random masking
        noise = torch.rand(batch_size, n_patches, device=patches.device)
        mask_indices = noise < mask_ratio  # (batch, n_patches)
        
        # Embed patches and mask token
        embedded_patches = self.patch_embedding(patches)
        mask_tokens = self.mask_token.expand(batch_size, n_patches, -1)
        
        # Replace masked positions with mask token
        masked_patches = torch.where(
            mask_indices.unsqueeze(-1),
            mask_tokens,
            embedded_patches
        )
        
        # Need to return patches in original space for reconstruction loss
        return masked_patches, mask_indices


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class NeuralPerturbationDataset(Dataset):
    """
    Dataset class for loading neural data from individual NPZ files.
    
    Each NPZ file should contain:
        - 'neural_data': array of shape (time_steps, n_neurons)
        - 'label': perturbation class index (optional, for supervised training)
    """
    def __init__(self, npz_files, supervised=True):
        """
        Args:
            npz_files: List of paths to NPZ files or directory containing NPZ files
            supervised: Whether labels are available
        """
        if isinstance(npz_files, (str, Path)):
            # If directory provided, get all NPZ files
            npz_dir = Path(npz_files)
            self.npz_files = sorted(list(npz_dir.glob("*.npz")))
        else:
            self.npz_files = [Path(f) for f in npz_files]
        
        self.supervised = supervised
        
        print(f"Found {len(self.npz_files)} NPZ files")
    
    def __len__(self):
        return len(self.npz_files)
    
    def __getitem__(self, idx):
        # Load NPZ file
        data = np.load(self.npz_files[idx])
        
        # Get neural data (time_steps, n_neurons)
        neural_data = data['neural_data']
        neural_data = torch.FloatTensor(neural_data)
        
        if self.supervised and 'label' in data:
            label = int(data['label'])
            return neural_data, label
        else:
            return neural_data, -1  # Return dummy label for unsupervised


# Training functions
def pretrain_masked(model, dataloader, optimizer, device, mask_ratio=0.15):
    """
    Pretrain based on masked prediction.
    
    Args:
        model: NeuralPerturbationTransformer with use_masking=True
        dataloader: DataLoader for neural data
        optimizer: torch optimizer
        device: torch device
        mask_ratio: proportion of patches to mask
    Returns:
        avg_loss: average reconstruction loss
    """
    model.train()
    total_loss = 0
    n_batches = 0
    
    for neural_data, _ in dataloader:
        neural_data = neural_data.to(device)
        
        optimizer.zero_grad()
        
        logits, reconstruction, mask_indices = model(neural_data, mask_ratio=mask_ratio)
        
        # Get original patches for reconstruction target
        patches = model.patchify(neural_data)
        
        # Compute reconstruction loss only on masked patches
        if mask_indices is not None:
            masked_patches = patches[mask_indices]
            reconstructed_patches = reconstruction[mask_indices]
            loss = F.mse_loss(reconstructed_patches, masked_patches)
        else:
            loss = torch.tensor(0.0, device=device)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


def train_supervised(model, dataloader, optimizer, device):
    """
    Supervised training directly on perturbation prediction.
    
    Args:
        model: NeuralPerturbationTransformer
        dataloader: DataLoader for neural data with labels
        optimizer: torch optimizer
        device: torch device
    Returns:
        avg_loss: average classification loss
        avg_accuracy: average classification accuracy
    """
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    for neural_data, labels in dataloader:
        neural_data = neural_data.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        logits, _, _ = model(neural_data, mask_ratio=0.0)
        
        loss = F.cross_entropy(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        predictions = logits.argmax(dim=-1)
        correct = (predictions == labels).sum().item()
        
        total_loss += loss.item()
        total_correct += correct
        total_samples += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_correct / total_samples
    
    return avg_loss, avg_accuracy


def evaluate(model, dataloader, device):
    """
    Evaluate model on validation/test data.
    
    Args:
        model: NeuralPerturbationTransformer
        dataloader: DataLoader for neural data with labels
        device: torch device
    Returns:
        avg_loss: average classification loss
        avg_accuracy: average classification accuracy
    """
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for neural_data, labels in dataloader:
            neural_data = neural_data.to(device)
            labels = labels.to(device)
            
            logits, _, _ = model(neural_data, mask_ratio=0.0)
            
            loss = F.cross_entropy(logits, labels)
            predictions = logits.argmax(dim=-1)
            correct = (predictions == labels).sum().item()
            
            total_loss += loss.item()
            total_correct += correct
            total_samples += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_correct / total_samples
    
    return avg_loss, avg_accuracy


# Example usage
if __name__ == "__main__":
    # Hyperparameters
    data_dir = "path/to/npz/files"  # Directory containing NPZ files
    batch_size = 16
    n_neurons = 100  # Should match your data
    patch_size = 50
    n_perturbations = 10000
    n_epochs = 10
    learning_rate = 1e-4
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create datasets
    print("\nLoading data...")
    train_dataset = NeuralPerturbationDataset(data_dir, supervised=True)
    
    # Create dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Infer time_steps from first sample
    sample_data, _ = train_dataset[0]
    time_steps = sample_data.shape[0]
    n_neurons = sample_data.shape[1]
    print(f"Data shape: ({time_steps} time steps, {n_neurons} neurons)")
    
    # Create model
    model = NeuralPerturbationTransformer(
        n_neurons=n_neurons,
        patch_size=patch_size,
        d_model=256,
        n_heads=8,
        n_layers=6,
        n_perturbations=n_perturbations,
        dropout=0.1,
        use_masking=True  # Set to True for masked pretraining
    ).to(device)
    
    print(f"\nModel has {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Option 1: Masked pretraining (self-supervised)
    print("\n" + "="*50)
    print("MASKED PRETRAINING")
    print("="*50)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(n_epochs):
        avg_loss = pretrain_masked(model, train_loader, optimizer, device, mask_ratio=0.15)
        print(f"Epoch {epoch+1}/{n_epochs}, Reconstruction Loss: {avg_loss:.4f}")
    
    # Option 2: Supervised training (or fine-tuning after pretraining)
    print("\n" + "="*50)
    print("SUPERVISED TRAINING")
    print("="*50)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(n_epochs):
        train_loss, train_acc = train_supervised(model, train_loader, optimizer, device)
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
    
    # Save model
    print("\nSaving model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'n_neurons': n_neurons,
        'patch_size': patch_size,
        'd_model': 256,
        'n_heads': 8,
        'n_layers': 6,
        'n_perturbations': n_perturbations,
    }, 'neural_perturbation_model.pt')
    print("Model saved to 'neural_perturbation_model.pt'")
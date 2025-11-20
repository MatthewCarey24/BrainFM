import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
        Convert neural data into patch tokens.
        
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


# Training functions
def pretrain_masked(model, neural_data, optimizer, mask_ratio=0.15):
    """
    pretrain based on masked prediction, perturbation prediction would be fine tuning head afterwards
    
    Args:
        model: NeuralPerturbationTransformer with use_masking=True
        neural_data: (batch, time, n_neurons) - can be unlabeled data
        optimizer: torch optimizer
        mask_ratio: proportion of patches to mask
    Returns:
        loss: reconstruction loss
    """
    model.train()
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
        loss = torch.tensor(0.0, device=neural_data.device)
    
    loss.backward()
    optimizer.step()
    
    return loss.item()


def train_supervised(model, neural_data, labels, optimizer):
    """
    Supervised training directly on perturbation prediction
    
    Args:
        model: NeuralPerturbationTransformer
        neural_data: (batch, time, n_neurons)
        labels: (batch,) - perturbation class indices
        optimizer: torch optimizer
    Returns:
        loss: classification loss
        accuracy: classification accuracy
    """
    model.train()
    optimizer.zero_grad()
    
    logits, _, _ = model(neural_data, mask_ratio=0.0)
    
    loss = F.cross_entropy(logits, labels)
    
    loss.backward()
    optimizer.step()
    
    # Calculate accuracy
    predictions = logits.argmax(dim=-1)
    accuracy = (predictions == labels).float().mean().item()
    
    return loss.item(), accuracy


# Example usage
if __name__ == "__main__":
    # Hyperparameters
    batch_size = 16
    n_neurons = 100
    time_steps = 1000
    patch_size = 50
    n_perturbations = 10000
    
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
    )
    
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    
    # Generate synthetic data
    neural_data = torch.randn(batch_size, time_steps, n_neurons)
    labels = torch.randint(0, n_perturbations, (batch_size,))
    
    # Option 1: Masked pretraining (self-supervised)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    print("\nMasked pretraining:")
    for epoch in range(3):
        loss = pretrain_masked(model, neural_data, optimizer, mask_ratio=0.15)
        print(f"Epoch {epoch+1}, Reconstruction Loss: {loss:.4f}")
    
    # Option 2: Supervised training (or fine-tuning after pretraining)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    print("\nSupervised training:")
    for epoch in range(3):
        loss, acc = train_supervised(model, neural_data, labels, optimizer)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}, Accuracy: {acc:.4f}")
    
    # Inference
    model.eval()
    with torch.no_grad():
        logits, _, _ = model(neural_data, mask_ratio=0.0)
        probabilities = F.softmax(logits, dim=-1)
        predictions = logits.argmax(dim=-1)
        print(f"\nPredictions: {predictions[:5]}")
        print(f"True labels: {labels[:5]}")
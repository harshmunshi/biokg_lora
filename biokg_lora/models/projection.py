"""
Projection layer to map KG embeddings to LLM token space.

Stage 2: Train this layer to align RotatE embeddings with LLM embeddings.
"""

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KGProjection(nn.Module):
    """
    Projects KG embeddings to LLM token embedding space.
    
    Architecture:
        KG embedding (256-dim) → MLP → LLM embedding (4096-dim)
    
    Training:
        Contrastive learning to align with LLM's token embeddings
    """
    
    def __init__(
        self,
        kg_embedding_dim: int = 256,
        lm_embedding_dim: int = 4096,
        hidden_dim: int = 1024,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_layernorm: bool = True,
    ):
        """
        Args:
            kg_embedding_dim: Input dimension (RotatE embeddings)
            lm_embedding_dim: Output dimension (LLM embeddings)
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            dropout: Dropout rate
            use_layernorm: Whether to use layer normalization
        """
        super().__init__()
        
        self.kg_embedding_dim = kg_embedding_dim
        self.lm_embedding_dim = lm_embedding_dim
        
        # Build MLP
        layers = []
        
        # Input layer
        layers.append(nn.Linear(kg_embedding_dim, hidden_dim))
        if use_layernorm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, lm_embedding_dim))
        
        self.projection = nn.Sequential(*layers)
        
        logger.info(f"Initialized KGProjection: {kg_embedding_dim} → "
                   f"{hidden_dim} (×{num_layers}) → {lm_embedding_dim}")
    
    def forward(self, kg_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Project KG embeddings to LLM space.
        
        Args:
            kg_embeddings: (B, kg_embedding_dim) or (kg_embedding_dim,)
        
        Returns:
            lm_embeddings: (B, lm_embedding_dim) or (lm_embedding_dim,)
        """
        return self.projection(kg_embeddings)


def contrastive_projection_loss(
    kg_embeddings: torch.Tensor,
    entity_names: list,
    lm_embeddings: torch.Tensor,
    lm_tokenizer,
    projection: KGProjection,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Contrastive loss for training projection layer.
    
    Goal: Align projected KG embeddings with LLM token embeddings.
    
    Args:
        kg_embeddings: (B, kg_dim) KG embeddings from RotatE
        entity_names: List[str] entity names (e.g., ["Thbd", "Bmp4"])
        lm_embeddings: LLM's token embedding matrix
        lm_tokenizer: LLM's tokenizer
        projection: KGProjection model
        temperature: Temperature for contrastive loss
    
    Returns:
        loss: Contrastive loss value
    """
    B = kg_embeddings.size(0)
    device = kg_embeddings.device
    
    # Project KG embeddings to LLM space
    projected = projection(kg_embeddings)  # (B, lm_dim)
    projected = F.normalize(projected, dim=-1)
    
    # Get LLM embeddings for entity names
    target_embeddings = []
    for name in entity_names:
        # Tokenize entity name (usually single token for gene symbols)
        token_ids = lm_tokenizer.encode(name, add_special_tokens=False)
        if len(token_ids) > 0:
            # Take first token's embedding
            token_emb = lm_embeddings.weight[token_ids[0]]
            target_embeddings.append(token_emb)
        else:
            # Fallback: use mean of all embeddings
            target_embeddings.append(lm_embeddings.weight.mean(dim=0))
    
    target_embeddings = torch.stack(target_embeddings)  # (B, lm_dim)
    target_embeddings = F.normalize(target_embeddings, dim=-1)
    
    # Compute similarity matrix
    sim_matrix = torch.matmul(projected, target_embeddings.T) / temperature  # (B, B)
    
    # Labels: diagonal (positive pairs)
    labels = torch.arange(B, device=device)
    
    # InfoNCE loss
    loss = F.cross_entropy(sim_matrix, labels)
    
    return loss


class EntityAugmentation(nn.Module):
    """
    Augment LLM token embeddings with KG information.
    
    Used during inference in BioKG-LoRA model.
    """
    
    def __init__(
        self,
        projection: KGProjection,
        fusion_method: str = "add",
        fusion_weight: float = 0.5,
    ):
        """
        Args:
            projection: Trained KGProjection layer
            fusion_method: "add", "concat", or "gated"
            fusion_weight: Weight for KG information (0 to 1)
        """
        super().__init__()
        
        self.projection = projection
        self.fusion_method = fusion_method
        self.fusion_weight = fusion_weight
        
        if fusion_method == "gated":
            # Learnable gate
            self.gate = nn.Sequential(
                nn.Linear(projection.lm_embedding_dim * 2, projection.lm_embedding_dim),
                nn.Sigmoid()
            )
    
    def forward(
        self,
        token_embeddings: torch.Tensor,
        kg_embeddings: Optional[torch.Tensor] = None,
        entity_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Augment token embeddings with KG information.
        
        Args:
            token_embeddings: (B, L, lm_dim) LLM token embeddings
            kg_embeddings: (B, L, kg_dim) KG embeddings for entities
                          (None for non-entity tokens)
            entity_mask: (B, L) mask indicating which tokens are entities
        
        Returns:
            augmented_embeddings: (B, L, lm_dim)
        """
        if kg_embeddings is None or entity_mask is None:
            return token_embeddings
        
        # Project KG embeddings
        B, L, _ = kg_embeddings.shape
        kg_projected = self.projection(
            kg_embeddings.reshape(-1, kg_embeddings.size(-1))
        ).reshape(B, L, -1)  # (B, L, lm_dim)
        
        if self.fusion_method == "add":
            # Weighted addition
            augmented = token_embeddings + self.fusion_weight * kg_projected * entity_mask.unsqueeze(-1)
        
        elif self.fusion_method == "concat":
            # This would require changing model architecture
            raise NotImplementedError("Concat fusion not implemented")
        
        elif self.fusion_method == "gated":
            # Gated fusion
            combined = torch.cat([token_embeddings, kg_projected], dim=-1)
            gate_values = self.gate(combined)
            augmented = gate_values * kg_projected + (1 - gate_values) * token_embeddings
            augmented = augmented * entity_mask.unsqueeze(-1) + token_embeddings * (~entity_mask).unsqueeze(-1)
        
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
        return augmented


if __name__ == "__main__":
    # Test projection layer
    projection = KGProjection(
        kg_embedding_dim=256,
        lm_embedding_dim=4096,
        hidden_dim=1024,
        num_layers=2,
    )
    
    # Dummy input
    kg_emb = torch.randn(32, 256)
    
    # Forward pass
    lm_emb = projection(kg_emb)
    print(f"Input shape: {kg_emb.shape}")
    print(f"Output shape: {lm_emb.shape}")
    
    # Test augmentation
    aug = EntityAugmentation(projection, fusion_method="add")
    
    token_emb = torch.randn(2, 10, 4096)
    kg_emb_seq = torch.randn(2, 10, 256)
    entity_mask = torch.tensor([[1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
                                 [0, 1, 0, 0, 1, 0, 0, 0, 0, 1]], dtype=torch.bool)
    
    augmented = aug(token_emb, kg_emb_seq, entity_mask)
    print(f"Augmented shape: {augmented.shape}")

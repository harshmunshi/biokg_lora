"""
RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space.

Reference: Sun et al. "RotatE: Knowledge Graph Embedding by Relational 
          Rotation in Complex Space." ICLR 2019.

Key Idea:
- Entities: h, t ∈ ℂ^d (complex-valued embeddings)
- Relations: r ∈ [0, 2π)^{d/2} (phase angles)
- Score: d(h ∘ r, t) = ||h ∘ r - t|| (distance after rotation)
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RotatEEmbedding(nn.Module):
    """
    RotatE embeddings for entities and relations.
    
    This can be used standalone or as part of a larger model.
    """
    
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 256,
        margin: float = 9.0,
        epsilon: float = 2.0,
    ):
        """
        Args:
            num_entities: Number of entities in KG
            num_relations: Number of relation types
            embedding_dim: Embedding dimension (must be even)
            margin: Margin for margin-based loss
            epsilon: Regularization for embedding initialization
        """
        super().__init__()
        
        assert embedding_dim % 2 == 0, "embedding_dim must be even for complex numbers"
        
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.epsilon = epsilon
        
        # Entity embeddings (complex-valued)
        # Stored as real and imaginary parts concatenated: (num_entities, embedding_dim)
        self.entity_embedding = nn.Embedding(
            num_entities,
            embedding_dim,
            max_norm=1.0  # Constrain to unit sphere
        )
        
        # Relation embeddings (phase angles on unit circle)
        # Stored as angles in [0, 2π): (num_relations, embedding_dim // 2)
        self.relation_embedding = nn.Embedding(
            num_relations,
            embedding_dim // 2
        )
        
        self._init_embeddings()
        
        logger.info(f"Initialized RotatE: {num_entities} entities, "
                   f"{num_relations} relations, dim={embedding_dim}")
    
    def _init_embeddings(self):
        """Initialize embeddings uniformly."""
        nn.init.uniform_(
            self.entity_embedding.weight,
            -self.epsilon / self.embedding_dim,
            self.epsilon / self.embedding_dim
        )
        nn.init.uniform_(
            self.relation_embedding.weight,
            -self.epsilon / (self.embedding_dim // 2),
            self.epsilon / (self.embedding_dim // 2)
        )
    
    def get_entity_embedding(self, entity_ids: torch.Tensor) -> torch.Tensor:
        """Get entity embeddings."""
        return self.entity_embedding(entity_ids)
    
    def forward(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute RotatE scores for triples.
        
        Args:
            head: (B,) head entity IDs
            relation: (B,) relation type IDs
            tail: (B,) tail entity IDs
        
        Returns:
            scores: (B,) RotatE scores (lower is better, higher probability)
        """
        return self.score_triples(head, relation, tail)
    
    def score_triples(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute RotatE scores for (head, relation, tail) triples.
        
        Lower score = higher probability that triple is true.
        
        Args:
            head: (B,) head entity IDs
            relation: (B,) relation type IDs
            tail: (B,) tail entity IDs
        
        Returns:
            scores: (B,) distances in complex space
        """
        # Get embeddings
        h = self.entity_embedding(head)  # (B, embedding_dim)
        r = self.relation_embedding(relation)  # (B, embedding_dim // 2)
        t = self.entity_embedding(tail)  # (B, embedding_dim)
        
        # Split complex embeddings into real and imaginary parts
        # h = [re_h, im_h], t = [re_t, im_t]
        re_h, im_h = torch.chunk(h, 2, dim=-1)  # Each (B, embedding_dim // 2)
        re_t, im_t = torch.chunk(t, 2, dim=-1)
        
        # Relation as rotation: convert angles to unit circle
        # r_phase ∈ [0, 2π)
        phase = r / (self.margin / torch.pi)  # Normalize to [0, 2π)
        re_r = torch.cos(phase)  # Real part of e^{iθ}
        im_r = torch.sin(phase)  # Imaginary part of e^{iθ}
        
        # Complex multiplication: h ∘ r = (h_real + i*h_imag) * (r_real + i*r_imag)
        # Result: (h_real*r_real - h_imag*r_imag) + i*(h_real*r_imag + h_imag*r_real)
        re_hr = re_h * re_r - im_h * im_r
        im_hr = re_h * im_r + im_h * re_r
        
        # Distance in complex space: ||h ∘ r - t||
        re_diff = re_hr - re_t
        im_diff = im_hr - im_t
        
        # L2 distance
        score = torch.sqrt(re_diff ** 2 + im_diff ** 2 + 1e-8).sum(dim=-1)
        
        return score


class RotatE(nn.Module):
    """
    Full RotatE model with training utilities.
    
    Includes methods for:
    - Negative sampling
    - Self-adversarial weighting
    - Link prediction evaluation
    """
    
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 256,
        margin: float = 9.0,
        epsilon: float = 2.0,
    ):
        super().__init__()
        
        self.rotate = RotatEEmbedding(
            num_entities=num_entities,
            num_relations=num_relations,
            embedding_dim=embedding_dim,
            margin=margin,
            epsilon=epsilon,
        )
        
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.margin = margin
    
    def forward(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
        mode: str = "single",
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            head: Head entity IDs
            relation: Relation type IDs
            tail: Tail entity IDs
            mode: "single", "head-batch", or "tail-batch"
        
        Returns:
            scores: Scores for the triples
        """
        if mode == "single":
            return self.rotate.score_triples(head, relation, tail)
        
        elif mode == "head-batch":
            # Score all possible heads for given (relation, tail)
            return self._score_all_heads(head, relation, tail)
        
        elif mode == "tail-batch":
            # Score all possible tails for given (head, relation)
            return self._score_all_tails(head, relation, tail)
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def _score_all_tails(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score all possible tails for given (head, relation) pairs.
        
        Used for evaluation.
        
        Args:
            head: (B,) head entity IDs
            relation: (B,) relation type IDs
            tail: (B,) true tail entity IDs (not used, just for signature)
        
        Returns:
            scores: (B, num_entities) scores for all possible tails
        """
        B = head.size(0)
        scores = []
        
        for i in range(B):
            # Get head and relation embeddings
            h = self.rotate.entity_embedding(head[i:i+1])  # (1, embedding_dim)
            r = self.rotate.relation_embedding(relation[i:i+1])  # (1, embedding_dim // 2)
            
            # Get all entity embeddings as candidates
            all_entities = self.rotate.entity_embedding.weight  # (num_entities, embedding_dim)
            
            # Score against all entities
            re_h, im_h = torch.chunk(h, 2, dim=-1)
            phase = r / (self.margin / torch.pi)
            re_r, im_r = torch.cos(phase), torch.sin(phase)
            
            # Rotate head
            re_hr = re_h * re_r - im_h * im_r
            im_hr = re_h * im_r + im_h * re_r
            
            # Distance to all entities
            re_t, im_t = torch.chunk(all_entities, 2, dim=-1)
            re_diff = re_hr - re_t  # (num_entities, embedding_dim // 2)
            im_diff = im_hr - im_t
            
            score = torch.sqrt(re_diff ** 2 + im_diff ** 2 + 1e-8).sum(dim=-1)
            scores.append(score)
        
        return torch.stack(scores)  # (B, num_entities)
    
    def _score_all_heads(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score all possible heads for given (relation, tail) pairs.
        
        Args:
            head: (B,) true head entity IDs (not used, just for signature)
            relation: (B,) relation type IDs
            tail: (B,) tail entity IDs
        
        Returns:
            scores: (B, num_entities) scores for all possible heads
        """
        B = tail.size(0)
        scores = []
        
        for i in range(B):
            # Get relation and tail embeddings
            r = self.rotate.relation_embedding(relation[i:i+1])
            t = self.rotate.entity_embedding(tail[i:i+1])
            
            # Get all entity embeddings as candidates
            all_entities = self.rotate.entity_embedding.weight
            
            # Need to find h such that h ∘ r ≈ t
            # Equivalent to: h ≈ t ∘ r^{-1}
            # r^{-1} = conjugate = (re_r, -im_r)
            re_t, im_t = torch.chunk(t, 2, dim=-1)
            phase = r / (self.margin / torch.pi)
            re_r, im_r = torch.cos(phase), torch.sin(phase)
            
            # Rotate tail by inverse relation
            re_t_rinv = re_t * re_r + im_t * im_r
            im_t_rinv = im_t * re_r - re_t * im_r
            
            # Distance from all entities
            re_h, im_h = torch.chunk(all_entities, 2, dim=-1)
            re_diff = re_h - re_t_rinv
            im_diff = im_h - im_t_rinv
            
            score = torch.sqrt(re_diff ** 2 + im_diff ** 2 + 1e-8).sum(dim=-1)
            scores.append(score)
        
        return torch.stack(scores)
    
    def get_entity_embeddings(self) -> torch.Tensor:
        """Get all entity embeddings."""
        return self.rotate.entity_embedding.weight.data
    
    def get_relation_embeddings(self) -> torch.Tensor:
        """Get all relation embeddings."""
        return self.rotate.relation_embedding.weight.data


def rotate_loss(
    model: RotatE,
    positive_sample: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    negative_sample: torch.Tensor,
    mode: str = "head-batch",
    alpha: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute RotatE loss with self-adversarial negative sampling.
    
    Reference: Sun et al. ICLR 2019, Equation (3)
    
    Args:
        model: RotatE model
        positive_sample: (head, relation, tail) tensor tuple
        negative_sample: (B, neg_size) negative entity IDs
        mode: "head-batch" or "tail-batch"
        alpha: Temperature for self-adversarial weighting
    
    Returns:
        loss: Total loss value
        metrics: Dict of loss components
    """
    head, relation, tail = positive_sample
    
    # Positive scores
    positive_score = model.rotate.score_triples(head, relation, tail)  # (B,)
    
    # Negative scores
    B, neg_size = negative_sample.size()
    
    if mode == "head-batch":
        # Corrupted heads
        negative_score = []
        for i in range(neg_size):
            score = model.rotate.score_triples(
                negative_sample[:, i], relation, tail
            )
            negative_score.append(score)
        negative_score = torch.stack(negative_score, dim=1)  # (B, neg_size)
    
    else:  # tail-batch
        # Corrupted tails
        negative_score = []
        for i in range(neg_size):
            score = model.rotate.score_triples(
                head, relation, negative_sample[:, i]
            )
            negative_score.append(score)
        negative_score = torch.stack(negative_score, dim=1)  # (B, neg_size)
    
    # Self-adversarial weighting
    # Give higher weight to hard negatives (low score = high probability)
    negative_weights = F.softmax(-negative_score * alpha, dim=1).detach()
    
    # Margin-based loss with sigmoid
    margin = model.margin
    
    # Positive: want score < margin (high probability)
    positive_loss = F.logsigmoid(margin - positive_score).mean()
    
    # Negative: want score > margin (low probability)
    negative_loss = (
        negative_weights * F.logsigmoid(negative_score - margin)
    ).sum(dim=1).mean()
    
    loss = -(positive_loss + negative_loss)
    
    # Regularization: keep embeddings on unit sphere
    reg_loss = (
        model.rotate.entity_embedding.weight.norm(p=2, dim=1).mean() +
        model.rotate.relation_embedding.weight.norm(p=2, dim=1).mean()
    )
    
    total_loss = loss + 0.01 * reg_loss
    
    metrics = {
        "loss": loss.item(),
        "positive_loss": positive_loss.item(),
        "negative_loss": negative_loss.item(),
        "reg_loss": reg_loss.item(),
        "positive_score_mean": positive_score.mean().item(),
        "negative_score_mean": negative_score.mean().item(),
    }
    
    return total_loss, metrics


if __name__ == "__main__":
    # Test RotatE model
    model = RotatE(num_entities=1000, num_relations=10, embedding_dim=256)
    
    # Dummy data
    head = torch.randint(0, 1000, (32,))
    relation = torch.randint(0, 10, (32,))
    tail = torch.randint(0, 1000, (32,))
    negative_sample = torch.randint(0, 1000, (32, 64))
    
    # Forward pass
    scores = model(head, relation, tail)
    print(f"Scores shape: {scores.shape}")
    
    # Loss computation
    loss, metrics = rotate_loss(
        model,
        (head, relation, tail),
        negative_sample,
        mode="tail-batch"
    )
    print(f"Loss: {loss.item():.4f}")
    print(f"Metrics: {metrics}")

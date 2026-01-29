"""
BioKG-LoRA: Knowledge Graph Enhanced LLMs for Clinical Reasoning

Main package for training and evaluating LLMs augmented with biological
knowledge graph embeddings via LoRA fine-tuning.
"""

__version__ = "0.1.0"
__author__ = "Pragya Research Team"

from biokg_lora.models.rotate import RotatE, RotatEEmbedding
from biokg_lora.models.projection import KGProjection
from biokg_lora.models.biokg_lora import BioKGLoRA
from biokg_lora.data.kg_builder import BiologicalKGBuilder, create_dummy_kg

__all__ = [
    "RotatE",
    "RotatEEmbedding",
    "KGProjection",
    "BioKGLoRA",
    "BiologicalKGBuilder",
    "create_dummy_kg",
]

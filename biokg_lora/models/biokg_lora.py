"""
BioKG-LoRA: Main model integrating LLM + KG + LoRA.

Combines:
- Base LLM (Llama-3-8B or Mistral-7B)
- RotatE KG embeddings (frozen)
- Projection layer (trainable)
- LoRA adapters (trainable)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from biokg_lora.models.projection import KGProjection, EntityAugmentation
from biokg_lora.models.entity_linker import EntityLinker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BioKGLoRA(nn.Module):
    """
    Knowledge Graph Enhanced Language Model with LoRA.
    
    Architecture:
        1. Base LLM (frozen)
        2. RotatE KG embeddings (frozen)
        3. Projection layer (trainable)
        4. LoRA adapters (trainable)
        5. Entity linker for augmentation
    """
    
    def __init__(
        self,
        base_model: str = "meta-llama/Meta-Llama-3-8B",
        kg_embeddings_path: Optional[str] = None,
        entity2id_path: Optional[str] = None,
        kg_embedding_dim: int = 256,
        lora_rank: int = 32,
        lora_alpha: int = 64,
        lora_dropout: float = 0.05,
        projection_hidden_dim: int = 1024,
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        device_map: str = "auto",
    ):
        """
        Args:
            base_model: HuggingFace model name or path
            kg_embeddings_path: Path to RotatE embeddings .pt file
            entity2id_path: Path to entity2id.json
            kg_embedding_dim: KG embedding dimension
            lora_rank: LoRA rank
            lora_alpha: LoRA alpha
            lora_dropout: LoRA dropout
            projection_hidden_dim: Hidden dim for projection MLP
            load_in_4bit: Use 4-bit quantization (QLoRA)
            load_in_8bit: Use 8-bit quantization
            device_map: Device mapping strategy
        """
        super().__init__()
        
        self.base_model_name = base_model
        self.kg_embedding_dim = kg_embedding_dim
        
        # Load tokenizer
        logger.info(f"Loading tokenizer from {base_model}...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base LLM
        logger.info(f"Loading base LLM from {base_model}...")
        self.base_llm = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            device_map=device_map,
            torch_dtype=torch.float16 if (load_in_4bit or load_in_8bit) else torch.float32,
        )
        
        # Freeze base model
        for param in self.base_llm.parameters():
            param.requires_grad = False
        
        lm_embedding_dim = self.base_llm.config.hidden_size
        
        # Load KG embeddings (frozen)
        if kg_embeddings_path is not None:
            logger.info(f"Loading KG embeddings from {kg_embeddings_path}...")
            self.kg_embeddings = torch.load(kg_embeddings_path, weights_only=False)
            if isinstance(self.kg_embeddings, dict):
                self.kg_embeddings = self.kg_embeddings["embeddings"]
            self.kg_embeddings = nn.Parameter(self.kg_embeddings, requires_grad=False)
        else:
            logger.warning("No KG embeddings provided. Using random embeddings.")
            self.kg_embeddings = nn.Parameter(
                torch.randn(1000, kg_embedding_dim),
                requires_grad=False
            )
        
        # Load entity2id mapping
        if entity2id_path is not None:
            import json
            with open(entity2id_path) as f:
                entity2id = json.load(f)
            self.entity_linker = EntityLinker(entity2id, use_scispacy=False)
        else:
            logger.warning("No entity2id provided. Entity linking disabled.")
            self.entity_linker = None
        
        # Projection layer (trainable)
        logger.info("Initializing projection layer...")
        self.projection = KGProjection(
            kg_embedding_dim=kg_embedding_dim,
            lm_embedding_dim=lm_embedding_dim,
            hidden_dim=projection_hidden_dim,
            num_layers=2,
            dropout=0.1,
        )
        
        # Entity augmentation module
        self.augmentation = EntityAugmentation(
            projection=self.projection,
            fusion_method="add",
            fusion_weight=0.3,
        )
        
        # Apply LoRA
        logger.info("Applying LoRA adapters...")
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.base_llm = get_peft_model(self.base_llm, lora_config)
        
        logger.info("BioKG-LoRA model initialized successfully!")
        self.print_trainable_parameters()
    
    def print_trainable_parameters(self):
        """Print number of trainable parameters."""
        trainable_params = 0
        all_params = 0
        
        for _, param in self.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        logger.info(
            f"Trainable params: {trainable_params:,} || "
            f"All params: {all_params:,} || "
            f"Trainable%: {100 * trainable_params / all_params:.2f}%"
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        entity_ids: Optional[torch.Tensor] = None,
        entity_mask: Optional[torch.Tensor] = None,
        use_kg_augmentation: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional KG augmentation.
        
        Args:
            input_ids: (B, L) token IDs
            attention_mask: (B, L) attention mask
            labels: (B, L) target token IDs for training
            entity_ids: (B, L) entity IDs for each token (or -1 for non-entities)
            entity_mask: (B, L) boolean mask for entity positions
            use_kg_augmentation: Whether to use KG augmentation
        
        Returns:
            Dict with:
                - loss: (,) loss value (if labels provided)
                - logits: (B, L, vocab_size) next-token predictions
        """
        B, L = input_ids.shape
        
        if use_kg_augmentation and entity_ids is not None and entity_mask is not None:
            # Get token embeddings
            inputs_embeds = self.base_llm.get_input_embeddings()(input_ids)
            
            # Get KG embeddings for entities
            # entity_ids: (B, L), values in [0, num_entities) or -1 for non-entities
            valid_entity_ids = entity_ids.clone()
            valid_entity_ids[valid_entity_ids == -1] = 0  # Placeholder for non-entities
            
            kg_embeds = self.kg_embeddings[valid_entity_ids]  # (B, L, kg_dim)
            
            # Augment embeddings
            inputs_embeds = self.augmentation(
                token_embeddings=inputs_embeds,
                kg_embeddings=kg_embeds,
                entity_mask=entity_mask,
            )
            
            # Forward through LLM with augmented embeddings
            outputs = self.base_llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
            )
        else:
            # Standard forward pass without KG augmentation
            outputs = self.base_llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
        
        return {
            "loss": outputs.loss if labels is not None else None,
            "logits": outputs.logits,
        }
    
    def generate(
        self,
        input_text: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        use_kg_augmentation: bool = True,
        **kwargs
    ) -> str:
        """
        Generate text with optional KG augmentation.
        
        Args:
            input_text: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            use_kg_augmentation: Whether to use KG
            **kwargs: Additional generation arguments
        
        Returns:
            Generated text
        """
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
        ).to(self.base_llm.device)
        
        # Link entities if augmentation is enabled
        entity_ids = None
        entity_mask = None
        
        if use_kg_augmentation and self.entity_linker is not None:
            entities = self.entity_linker.link_entities(input_text)
            # TODO: Map character positions to token positions
            # For now, disable augmentation during generation
            logger.warning("KG augmentation during generation not fully implemented")
        
        # Generate
        with torch.no_grad():
            outputs = self.base_llm.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return generated_text
    
    def save_pretrained(self, save_directory: str):
        """Save model weights and configuration."""
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        
        # Save LoRA adapters
        self.base_llm.save_pretrained(save_directory / "lora_adapters")
        
        # Save projection layer
        torch.save(
            self.projection.state_dict(),
            save_directory / "projection.pt"
        )
        
        logger.info(f"Model saved to {save_directory}")
    
    def load_pretrained(self, load_directory: str):
        """Load model weights."""
        load_directory = Path(load_directory)
        
        # Load projection layer
        self.projection.load_state_dict(
            torch.load(load_directory / "projection.pt", weights_only=True)
        )
        
        logger.info(f"Model loaded from {load_directory}")


def create_dummy_biokg_lora(
    num_entities: int = 1000,
    kg_embedding_dim: int = 256,
) -> BioKGLoRA:
    """
    Create a dummy BioKG-LoRA model for testing.
    
    Uses a small model and random embeddings.
    """
    # Use a small model for testing
    model = BioKGLoRA(
        base_model="gpt2",  # Small model for testing
        kg_embeddings_path=None,  # Will use random embeddings
        entity2id_path=None,
        kg_embedding_dim=kg_embedding_dim,
        lora_rank=8,
        lora_alpha=16,
        projection_hidden_dim=256,
        load_in_4bit=False,
        load_in_8bit=False,
        device_map="cpu",
    )
    
    return model


if __name__ == "__main__":
    # Test with dummy model
    logger.info("Creating dummy BioKG-LoRA model...")
    
    model = create_dummy_biokg_lora(
        num_entities=100,
        kg_embedding_dim=256,
    )
    
    # Test forward pass
    input_ids = torch.randint(0, 1000, (2, 10))
    attention_mask = torch.ones(2, 10)
    entity_ids = torch.randint(-1, 100, (2, 10))
    entity_mask = (entity_ids >= 0)
    
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        entity_ids=entity_ids,
        entity_mask=entity_mask,
    )
    
    print(f"Output logits shape: {outputs['logits'].shape}")
    
    # Test generation
    generated = model.generate(
        "What is the function of gene Thbd?",
        max_new_tokens=50,
        use_kg_augmentation=False,
    )
    print(f"\nGenerated: {generated}")

#!/usr/bin/env python3
"""
Stage 3: LoRA Fine-tuning with KG Augmentation

Train LoRA adapters on biological QA task with KG-augmented embeddings.

Duration: 4-6 hours on 1× GPU (A100 or RTX 4090)

Usage:
    python scripts/train_lora.py \
        --base_model "meta-llama/Llama-3-8B" \
        --entity_embeddings checkpoints/stage1/entity_embeddings.pt \
        --entity2id_path data/kg/entity2id.json \
        --qa_dataset data/qa_dataset.json \
        --output_dir checkpoints/stage3 \
        --num_epochs 3
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm
import wandb

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KGProjectionLayer(nn.Module):
    """Project KG embeddings to LLM embedding space."""
    
    def __init__(self, kg_dim: int = 256, lm_dim: int = 4096):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(kg_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, lm_dim),
            nn.LayerNorm(lm_dim),
        )
    
    def forward(self, kg_emb):
        return self.proj(kg_emb)


class SimpleEntityLinker:
    """Simple rule-based entity linker."""
    
    def __init__(self, entity2id: Dict[str, int]):
        self.entity2id = entity2id
        self.lower_to_entity = {e.lower(): e for e in entity2id.keys()}
    
    def link_entities(self, text: str) -> List[Tuple[str, int, int]]:
        """Find entities in text. Returns (entity_name, start, end)."""
        entities_found = []
        text_lower = text.lower()
        
        for entity_lower, entity_name in self.lower_to_entity.items():
            start = 0
            while True:
                start = text_lower.find(entity_lower, start)
                if start == -1:
                    break
                end = start + len(entity_lower)
                entities_found.append((entity_name, start, end))
                start = end
        
        return entities_found


class BioQADataset(Dataset):
    """Dataset for biological QA with entity annotations."""
    
    def __init__(
        self,
        qa_data: List[Dict],
        tokenizer,
        entity_linker: SimpleEntityLinker,
        max_length: int = 512
    ):
        self.qa_data = qa_data
        self.tokenizer = tokenizer
        self.entity_linker = entity_linker
        self.max_length = max_length
    
    def __len__(self):
        return len(self.qa_data)
    
    def __getitem__(self, idx):
        qa = self.qa_data[idx]
        
        # Format as instruction-following
        text = f"Question: {qa['question']}\nAnswer: {qa['answer']}"
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Find entities in text
        entities = self.entity_linker.link_entities(text)
        
        # Convert char positions to token positions (approximate)
        entity_token_spans = []
        for entity_name, char_start, char_end in entities:
            # Simplified - find token spans
            tokens = self.tokenizer.tokenize(text[:char_start])
            token_start = len(tokens)
            tokens_entity = self.tokenizer.tokenize(text[char_start:char_end])
            token_end = token_start + len(tokens_entity)
            
            if token_end <= self.max_length:
                entity_token_spans.append((entity_name, token_start, token_end))
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'entities': entity_token_spans
        }


class BioKGLoRA(nn.Module):
    """KG-augmented LLM with LoRA."""
    
    def __init__(
        self,
        base_model_name: str,
        entity_embeddings: torch.Tensor,
        entity2id: Dict[str, int],
        lora_rank: int = 32,
        lora_alpha: int = 64,
        lora_dropout: float = 0.05,
        kg_weight: float = 0.3,
        use_4bit: bool = True
    ):
        super().__init__()
        
        self.entity2id = entity2id
        self.kg_weight = kg_weight
        
        # Load base model with 4-bit quantization
        logger.info(f"Loading base model: {base_model_name}")
        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            self.base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
            self.base_model = prepare_model_for_kbit_training(self.base_model)
        else:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                device_map="auto",
                trust_remote_code=True
            )
        
        # Register KG embeddings (frozen)
        self.register_buffer('entity_embeddings', entity_embeddings)
        
        # Projection layer
        kg_dim = entity_embeddings.shape[1]
        lm_dim = self.base_model.config.hidden_size
        logger.info(f"Creating projection layer: {kg_dim} → {lm_dim}")
        self.kg_projection = KGProjectionLayer(kg_dim=kg_dim, lm_dim=lm_dim)
        
        # Add LoRA
        logger.info(f"Adding LoRA adapters (rank={lora_rank})")
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.base_model = get_peft_model(self.base_model, lora_config)
        self.base_model.print_trainable_parameters()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        entity_spans: Optional[List[List[Tuple[str, int, int]]]] = None,
        labels: Optional[torch.Tensor] = None
    ):
        """
        Forward with KG augmentation.
        
        Args:
            input_ids: (B, L)
            attention_mask: (B, L)
            entity_spans: List of (entity_name, start_token, end_token) per batch
            labels: (B, L) for language modeling loss
        """
        # Get token embeddings
        embeddings = self.base_model.get_input_embeddings()
        token_embeds = embeddings(input_ids)
        
        # Augment with KG embeddings
        if entity_spans is not None:
            device = token_embeds.device
            for batch_idx, spans in enumerate(entity_spans):
                for entity_name, start, end in spans:
                    if entity_name in self.entity2id:
                        entity_id = self.entity2id[entity_name]
                        kg_emb = self.entity_embeddings[entity_id].to(device)
                        
                        # Project to LM space
                        kg_proj = self.kg_projection(kg_emb)
                        
                        # Fuse with weighted addition
                        if start < end and end <= token_embeds.size(1):
                            orig_embeds = token_embeds[batch_idx, start:end].mean(dim=0)
                            fused = (1 - self.kg_weight) * orig_embeds + self.kg_weight * kg_proj
                            token_embeds[batch_idx, start:end] = fused.unsqueeze(0).repeat(end - start, 1)
        
        # Forward through model
        outputs = self.base_model(
            inputs_embeds=token_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        
        return outputs


def train_epoch(
    model: BioKGLoRA,
    dataloader: DataLoader,
    optimizer,
    scheduler,
    device: str,
    epoch: int
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        entity_spans = batch['entities']
        
        # Labels for language modeling
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # Ignore padding
        
        # Forward
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            entity_spans=entity_spans,
            labels=labels
        )
        
        loss = outputs.loss
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    return total_loss / len(dataloader)


def evaluate(
    model: BioKGLoRA,
    dataloader: DataLoader,
    device: str
) -> float:
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            entity_spans = batch['entities']
            
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                entity_spans=entity_spans,
                labels=labels
            )
            
            total_loss += outputs.loss.item()
    
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description="Stage 3: LoRA Training")
    
    # Model
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3-8B",
                       help="Base LLM model")
    parser.add_argument("--use_4bit", action="store_true", default=True,
                       help="Use 4-bit quantization (QLoRA)")
    
    # Data
    parser.add_argument("--entity_embeddings", type=str, required=True,
                       help="Path to entity embeddings from Stage 1")
    parser.add_argument("--entity2id_path", type=str, required=True,
                       help="Path to entity2id.json")
    parser.add_argument("--qa_dataset", type=str, required=True,
                       help="Path to QA dataset JSON")
    
    # LoRA
    parser.add_argument("--lora_rank", type=int, default=32,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=64,
                       help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                       help="LoRA dropout")
    
    # KG Augmentation
    parser.add_argument("--kg_weight", type=float, default=0.3,
                       help="Weight for KG embedding fusion (alpha)")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                       help="Gradient accumulation")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                       help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of epochs")
    parser.add_argument("--warmup_steps", type=int, default=100,
                       help="Warmup steps")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Max sequence length")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="checkpoints/stage3",
                       help="Output directory")
    parser.add_argument("--use_wandb", action="store_true",
                       help="Use Weights & Biases logging")
    
    args = parser.parse_args()
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.use_wandb:
        wandb.init(project="biokg-lora", config=vars(args))
    
    # Load data
    logger.info("Loading data...")
    entity_embeddings = torch.load(args.entity_embeddings, weights_only=False)
    logger.info(f"  Entity embeddings: {entity_embeddings.shape}")
    
    with open(args.entity2id_path, 'r') as f:
        entity2id = json.load(f)
    logger.info(f"  Entity mappings: {len(entity2id)}")
    
    with open(args.qa_dataset, 'r') as f:
        qa_data = json.load(f)
    logger.info(f"  Train: {len(qa_data['train'])} QA pairs")
    logger.info(f"  Val: {len(qa_data['val'])} QA pairs")
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create datasets
    logger.info("Creating datasets...")
    entity_linker = SimpleEntityLinker(entity2id)
    
    train_dataset = BioQADataset(
        qa_data['train'],
        tokenizer,
        entity_linker,
        max_length=args.max_length
    )
    val_dataset = BioQADataset(
        qa_data['val'],
        tokenizer,
        entity_linker,
        max_length=args.max_length
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0  # 0 for compatibility
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Create model
    logger.info("Creating BioKG-LoRA model...")
    model = BioKGLoRA(
        base_model_name=args.base_model,
        entity_embeddings=entity_embeddings,
        entity2id=entity2id,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        kg_weight=args.kg_weight,
        use_4bit=args.use_4bit
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01
    )
    
    total_steps = len(train_loader) * args.num_epochs // args.gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    logger.info("="*60)
    logger.info("Starting LoRA Training")
    logger.info("="*60)
    
    best_val_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, epoch + 1)
        val_loss = evaluate(model, val_loader, device)
        
        logger.info(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        logger.info(f"  Train Loss: {train_loss:.4f}")
        logger.info(f"  Val Loss: {val_loss:.4f}")
        
        if args.use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss
            })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            logger.info(f"  ✓ New best model! Saving...")
            
            # Save LoRA adapters
            model.base_model.save_pretrained(output_dir / "lora_adapters")
            
            # Save projection layer
            torch.save(
                model.kg_projection.state_dict(),
                output_dir / "projection_layer.pt"
            )
            
            # Save config
            config = {
                "base_model": args.base_model,
                "lora_rank": args.lora_rank,
                "lora_alpha": args.lora_alpha,
                "kg_weight": args.kg_weight,
                "best_val_loss": best_val_loss
            }
            with open(output_dir / "config.json", 'w') as f:
                json.dump(config, f, indent=2)
    
    logger.info("\n" + "="*60)
    logger.info("Training Complete!")
    logger.info("="*60)
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Model saved to: {output_dir}")
    logger.info("\nNext: Test the model with demo.py")


if __name__ == "__main__":
    main()

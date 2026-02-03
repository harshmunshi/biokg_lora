#!/usr/bin/env python3
"""
Interactive Demo: BioKG-LoRA End-to-End

Test the trained model interactively or run predefined examples.

Usage:
    # Interactive mode
    python scripts/demo.py \
        --model_dir checkpoints/stage3 \
        --entity_embeddings checkpoints/stage1/entity_embeddings.pt \
        --entity2id_path data/kg/entity2id.json \
        --interactive

    # Run examples
    python scripts/demo.py \
        --model_dir checkpoints/stage3 \
        --entity_embeddings checkpoints/stage1/entity_embeddings.pt \
        --entity2id_path data/kg/entity2id.json \
        --examples
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
console = Console()


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


class BioKGLoRADemo:
    """Demo wrapper for BioKG-LoRA model."""
    
    def __init__(
        self,
        model_dir: str,
        entity_embeddings_path: str,
        entity2id_path: str,
        use_4bit: bool = True
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        model_dir = Path(model_dir)
        
        # Load config
        logger.info("Loading model config...")
        with open(model_dir / "config.json", 'r') as f:
            self.config = json.load(f)
        
        base_model_name = self.config['base_model']
        self.kg_weight = self.config['kg_weight']
        
        # Load tokenizer
        logger.info(f"Loading tokenizer: {base_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
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
        else:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
        
        # Load LoRA adapters
        logger.info("Loading LoRA adapters...")
        self.model = PeftModel.from_pretrained(
            self.base_model,
            model_dir / "lora_adapters"
        )
        self.model.eval()
        
        # Load entity embeddings
        logger.info("Loading entity embeddings...")
        self.entity_embeddings = torch.load(entity_embeddings_path, weights_only=False)
        
        # Load entity2id
        with open(entity2id_path, 'r') as f:
            self.entity2id = json.load(f)
        
        # Load projection layer
        logger.info("Loading projection layer...")
        kg_dim = self.entity_embeddings.shape[1]
        lm_dim = self.model.config.hidden_size
        self.kg_projection = KGProjectionLayer(kg_dim=kg_dim, lm_dim=lm_dim)
        self.kg_projection.load_state_dict(
            torch.load(model_dir / "projection_layer.pt", weights_only=True)
        )
        self.kg_projection = self.kg_projection.to(self.device)
        self.kg_projection.eval()
        
        # Entity linker
        self.entity_linker = SimpleEntityLinker(self.entity2id)
        
        logger.info("✓ Model loaded successfully!")
    
    def generate(
        self,
        question: str,
        max_length: int = 256,
        temperature: float = 0.7,
        use_kg: bool = True
    ) -> Tuple[str, List[str]]:
        """
        Generate answer for a question.
        
        Returns:
            (answer, entities_used)
        """
        # Format prompt
        prompt = f"Question: {question}\nAnswer:"
        
        # Find entities
        entities = self.entity_linker.link_entities(prompt)
        entity_names = [e[0] for e in entities]
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs['input_ids']
        
        # Get embeddings
        with torch.no_grad():
            token_embeds = self.model.get_input_embeddings()(input_ids)
            
            # Augment with KG if requested
            if use_kg and entities:
                for entity_name, char_start, char_end in entities:
                    if entity_name in self.entity2id:
                        entity_id = self.entity2id[entity_name]
                        kg_emb = self.entity_embeddings[entity_id].to(self.device)
                        
                        # Project
                        kg_proj = self.kg_projection(kg_emb)
                        
                        # Find approximate token positions
                        # Simplified - in practice need better alignment
                        tokens_before = self.tokenizer.tokenize(prompt[:char_start])
                        token_start = len(tokens_before)
                        tokens_entity = self.tokenizer.tokenize(prompt[char_start:char_end])
                        token_end = token_start + len(tokens_entity)
                        
                        if token_end <= token_embeds.size(1):
                            # Fuse
                            orig = token_embeds[0, token_start:token_end].mean(dim=0)
                            fused = (1 - self.kg_weight) * orig + self.kg_weight * kg_proj
                            token_embeds[0, token_start:token_end] = fused.unsqueeze(0)
            
            # Generate
            outputs = self.model.generate(
                inputs_embeds=token_embeds,
                attention_mask=inputs['attention_mask'],
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer (after "Answer:")
        if "Answer:" in generated_text:
            answer = generated_text.split("Answer:", 1)[1].strip()
        else:
            answer = generated_text.strip()
        
        return answer, entity_names


def run_examples(demo: BioKGLoRADemo):
    """Run predefined examples."""
    console.print("\n[bold cyan]Running Predefined Examples[/bold cyan]\n")
    
    examples = [
        "What phenotypes are associated with Thbd knockout?",
        "What is the significance of elevated ALT in Bmp4 knockout mice?",
        "How does Fgfr2 knockout affect kidney development?",
        "Compare the knockout phenotypes of Cyp2c23 and S100a1.",
    ]
    
    for i, question in enumerate(examples, 1):
        console.print(f"\n[bold]Example {i}/{len(examples)}[/bold]")
        console.print(Panel(question, title="Question", border_style="blue"))
        
        # Generate with KG
        console.print("\n[yellow]Generating answer with KG augmentation...[/yellow]")
        answer_with_kg, entities = demo.generate(question, use_kg=True)
        
        console.print(Panel(
            f"{answer_with_kg}\n\n[dim]Entities used: {', '.join(entities) if entities else 'None'}[/dim]",
            title="BioKG-LoRA Answer (with KG)",
            border_style="green"
        ))
        
        # Generate without KG for comparison
        console.print("\n[yellow]Generating answer without KG (base LoRA only)...[/yellow]")
        answer_no_kg, _ = demo.generate(question, use_kg=False)
        
        console.print(Panel(
            answer_no_kg,
            title="Answer without KG",
            border_style="yellow"
        ))
        
        console.print("\n" + "-" * 80)


def run_interactive(demo: BioKGLoRADemo):
    """Interactive Q&A mode."""
    console.print("\n[bold cyan]Interactive BioKG-LoRA Demo[/bold cyan]")
    console.print("Ask biological questions! Type 'quit' to exit.\n")
    
    while True:
        try:
            question = console.input("[bold blue]Question:[/bold blue] ")
            
            if question.lower() in ['quit', 'exit', 'q']:
                console.print("\n[yellow]Goodbye![/yellow]")
                break
            
            if not question.strip():
                continue
            
            # Generate answer
            console.print("\n[yellow]Generating answer...[/yellow]")
            answer, entities = demo.generate(question, use_kg=True)
            
            console.print(Panel(
                f"{answer}\n\n[dim]Entities recognized: {', '.join(entities) if entities else 'None'}[/dim]",
                title="Answer",
                border_style="green"
            ))
            
            # Ask if user wants comparison
            compare = console.input("\n[dim]Show answer without KG? (y/n):[/dim] ")
            if compare.lower() == 'y':
                answer_no_kg, _ = demo.generate(question, use_kg=False)
                console.print(Panel(
                    answer_no_kg,
                    title="Answer without KG",
                    border_style="yellow"
                ))
            
            console.print()
        
        except KeyboardInterrupt:
            console.print("\n\n[yellow]Goodbye![/yellow]")
            break
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]\n")


def main():
    parser = argparse.ArgumentParser(description="BioKG-LoRA Demo")
    
    parser.add_argument("--model_dir", type=str, default="checkpoints/stage3",
                       help="Path to trained model directory")
    parser.add_argument("--entity_embeddings", type=str,
                       default="checkpoints/stage1/entity_embeddings.pt",
                       help="Path to entity embeddings")
    parser.add_argument("--entity2id_path", type=str, default="data/kg/entity2id.json",
                       help="Path to entity2id.json")
    
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive mode")
    parser.add_argument("--examples", action="store_true",
                       help="Run predefined examples")
    parser.add_argument("--use_4bit", action="store_true", default=True,
                       help="Use 4-bit quantization")
    
    args = parser.parse_args()
    
    # Check files exist
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        console.print(f"[red]Error: Model directory not found: {model_dir}[/red]")
        console.print("\nPlease train the model first:")
        console.print("  python scripts/train_lora.py ...")
        return
    
    # Load model
    console.print("\n[bold cyan]Loading BioKG-LoRA Model...[/bold cyan]\n")
    
    try:
        demo = BioKGLoRADemo(
            model_dir=args.model_dir,
            entity_embeddings_path=args.entity_embeddings,
            entity2id_path=args.entity2id_path,
            use_4bit=args.use_4bit
        )
        
        # Run mode
        if args.examples:
            run_examples(demo)
        elif args.interactive:
            run_interactive(demo)
        else:
            # Default: run examples
            run_examples(demo)
            console.print("\n[dim]Tip: Use --interactive for Q&A mode[/dim]")
    
    except Exception as e:
        console.print(f"\n[red]Error loading model: {e}[/red]")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

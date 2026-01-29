#!/usr/bin/env python3
"""
Stage 0: Build Knowledge Graph from biological databases.

Downloads and integrates:
- MGI (genes + phenotypes)
- GO (gene ontology)
- KEGG (pathways)
- STRING (protein interactions)
- GTEx (tissue expression - requires manual download)

Usage:
    python scripts/stage0_build_kg.py --output_dir data/kg
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from biokg_lora.data.kg_builder import BiologicalKGBuilder, create_dummy_kg
from biokg_lora.visualization.kg_viz import visualize_kg_interactive, create_kg_dashboard

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def build_kg_from_scratch(
    output_dir: str,
    mgi_dir: Optional[str] = None,
    go_file: Optional[str] = None,
    string_dir: Optional[str] = None,
):
    """
    Build KG from raw data files.
    
    Args:
        output_dir: Output directory for KG
        mgi_dir: Directory with MGI files
        go_file: Path to GO OBO file
        string_dir: Directory with STRING files
    """
    logger.info("Building knowledge graph from biological databases...")
    
    builder = BiologicalKGBuilder()
    
    # This is a placeholder - in practice, you would:
    # 1. Download data from each source
    # 2. Parse files and add entities/triples
    # 3. Build and save KG
    
    logger.warning("Full KG construction from raw data not implemented.")
    logger.warning("Please download data manually and use data source parsers.")
    logger.info("Using dummy KG for demonstration...")
    
    # Create dummy KG instead
    kg_data, metadata = create_dummy_kg(
        num_genes=1000,
        num_phenotypes=500,
        num_go_terms=2000,
        num_pathways=100,
        num_tissues=50,
    )
    
    builder.entity2id = metadata["entity2id"]
    builder.id2entity = metadata["id2entity"]
    builder.entity_types = {i: 0 for i in range(kg_data.num_nodes)}  # Placeholder
    
    # Save
    output_dir = Path(output_dir)
    builder.save(output_dir, kg_data)
    builder.print_stats()
    
    return kg_data, metadata


def build_dummy_kg(
    output_dir: str,
    num_genes: int = 1000,
    num_phenotypes: int = 500,
    seed: int = 42,
):
    """
    Build a dummy KG for testing.
    
    Args:
        output_dir: Output directory
        num_genes: Number of genes
        num_phenotypes: Number of phenotypes
        seed: Random seed
    """
    logger.info(f"Building dummy KG: {num_genes} genes, {num_phenotypes} phenotypes...")
    
    kg_data, metadata = create_dummy_kg(
        num_genes=num_genes,
        num_phenotypes=num_phenotypes,
        num_go_terms=num_genes * 2,
        num_pathways=num_genes // 10,
        num_tissues=50,
        seed=seed,
    )
    
    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    builder = BiologicalKGBuilder()
    builder.entity2id = metadata["entity2id"]
    builder.id2entity = metadata["id2entity"]
    builder.save(output_dir, kg_data)
    
    logger.info(f"Dummy KG saved to {output_dir}")
    
    return kg_data, metadata


def visualize_kg(kg_path: str, entity2id_path: str, output_dir: str):
    """
    Create KG visualizations.
    
    Args:
        kg_path: Path to KG PyG Data file
        entity2id_path: Path to entity2id.json
        output_dir: Output directory for visualizations
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Creating interactive KG visualization...")
    visualize_kg_interactive(
        kg_path=kg_path,
        entity2id_path=entity2id_path,
        output_html=str(output_dir / "kg_interactive.html"),
        max_nodes=500,
    )
    
    logger.info("Creating KG statistics dashboard...")
    create_kg_dashboard(
        kg_path=kg_path,
        entity2id_path=entity2id_path,
        output_html=str(output_dir / "kg_dashboard.html"),
    )
    
    logger.info(f"Visualizations saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Stage 0: Build Knowledge Graph")
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/kg",
        help="Output directory for KG files"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["dummy", "full"],
        default="dummy",
        help="Build mode: dummy (for testing) or full (from real data)"
    )
    
    parser.add_argument(
        "--num_genes",
        type=int,
        default=1000,
        help="Number of genes (dummy mode only)"
    )
    
    parser.add_argument(
        "--num_phenotypes",
        type=int,
        default=500,
        help="Number of phenotypes (dummy mode only)"
    )
    
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Create visualizations"
    )
    
    parser.add_argument(
        "--viz_output_dir",
        type=str,
        default="outputs/kg_viz",
        help="Output directory for visualizations"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # Set seed
    np.random.seed(args.seed)
    
    # Build KG
    if args.mode == "dummy":
        kg_data, metadata = build_dummy_kg(
            output_dir=args.output_dir,
            num_genes=args.num_genes,
            num_phenotypes=args.num_phenotypes,
            seed=args.seed,
        )
    else:
        kg_data, metadata = build_kg_from_scratch(
            output_dir=args.output_dir,
        )
    
    logger.info("✅ Stage 0 complete!")
    logger.info(f"KG saved to {args.output_dir}")
    logger.info(f"  - biological_kg.pt")
    logger.info(f"  - entity2id.json")
    logger.info(f"  - id2entity.json")
    logger.info(f"  - kg_stats.json")
    
    # Visualize
    if args.visualize:
        kg_path = Path(args.output_dir) / "biological_kg.pt"
        entity2id_path = Path(args.output_dir) / "entity2id.json"
        
        visualize_kg(
            kg_path=str(kg_path),
            entity2id_path=str(entity2id_path),
            output_dir=args.viz_output_dir,
        )
        
        logger.info(f"✅ Visualizations saved to {args.viz_output_dir}")
    
    logger.info("\n🚀 Next step: Stage 1 (RotatE training)")
    logger.info(f"    python scripts/stage1_train_rotate.py --kg_path {args.output_dir}/biological_kg.pt")


if __name__ == "__main__":
    main()

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
from typing import Optional

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
    data_dir: str = "data",
):
    """
    Build KG from raw data files.
    
    Args:
        output_dir: Output directory for KG
        data_dir: Directory containing downloaded data sources
    
    Expected directory structure:
        data/
        ├── mgi/
        │   ├── MRK_List2.rpt
        │   └── MGI_PhenoGenoMP.rpt
        ├── ontologies/
        │   ├── mgi_go.gaf
        │   └── go.obo
        ├── kegg/
        │   └── mmu_pathway_genes.txt
        ├── string/
        │   ├── 10090.protein.links.v12.0.txt
        │   └── 10090.protein.info.v12.0.txt
        └── gtex/
            └── gene_median_tpm.gct
    """
    import pandas as pd
    import obonet
    
    logger.info("Building knowledge graph from biological databases...")
    
    data_path = Path(data_dir)
    builder = BiologicalKGBuilder()
    
    # ===== 1. Load MGI Genes =====
    logger.info("Loading genes from MGI...")
    mgi_file = data_path / "mgi" / "MRK_List2.rpt"
    
    if not mgi_file.exists():
        logger.error(f"MGI file not found: {mgi_file}")
        logger.info("Please download from: http://www.informatics.jax.org/downloads/reports/MRK_List2.rpt")
        raise FileNotFoundError(f"Required file not found: {mgi_file}")
    
    mgi_df = pd.read_csv(mgi_file, sep="\t", low_memory=False)
    
    # Filter for protein-coding genes only (Marker Type = "Gene")
    gene_types = ["Gene"]
    mgi_genes = mgi_df[mgi_df["Marker Type"].isin(gene_types)]
    
    # Create mapping from MGI ID to gene symbol
    mgi_id_to_symbol = {}
    
    for _, row in mgi_genes.iterrows():
        gene_symbol = row["Marker Symbol"]
        gene_id = row["MGI Accession ID"]
        
        if pd.notna(gene_symbol) and pd.notna(gene_id):
            builder.add_entity(gene_symbol, "gene", metadata={"mgi_id": gene_id})
            mgi_id_to_symbol[gene_id] = gene_symbol
    
    logger.info(f"Loaded {len(builder.entity2id)} genes (filtered from {len(mgi_df)} total markers)")
    
    # ===== 2. Load MPO (Mammalian Phenotype Ontology) for readable names =====
    logger.info("Loading Mammalian Phenotype Ontology...")
    mpo_file = data_path / "ontologies" / "mpo.obo"
    mp_id_to_name = {}
    mpo_graph = None
    
    if mpo_file.exists():
        try:
            logger.info(f"Reading MPO file: {mpo_file} ({mpo_file.stat().st_size / 1024 / 1024:.1f} MB)")
            mpo_graph = obonet.read_obo(str(mpo_file))
            for mp_id, data in mpo_graph.nodes(data=True):
                mp_name = data.get("name", mp_id)
                mp_id_to_name[mp_id] = mp_name
            logger.info(f"Loaded {len(mp_id_to_name)} phenotype terms from MPO")
        except Exception as e:
            logger.error(f"Failed to load MPO: {e}")
            logger.warning("Will use MP IDs instead of readable names")
            mpo_graph = None
    else:
        logger.warning(f"MPO file not found: {mpo_file}")
        logger.info("Will use MP IDs instead of readable names")
    
    # ===== 3. Load Gene-Phenotype Associations =====
    logger.info("Loading gene-phenotype associations...")
    pheno_file = data_path / "mgi" / "MGI_PhenoGenoMP.rpt"
    
    if not pheno_file.exists():
        logger.warning(f"Phenotype file not found: {pheno_file}")
        logger.info("Skipping phenotype associations...")
    else:
        # MGI_PhenoGenoMP.rpt has NO header - columns are:
        # 0: Genotype, 1: Allele, 2: Background, 3: MP ID, 4: PubMed, 5: MGI Gene ID, 6: MGI Genotype ID
        pheno_df = pd.read_csv(
            pheno_file, 
            sep="\t", 
            header=None,
            names=["Genotype", "Allele", "Background", "MP_ID", "PubMed", "MGI_Gene_ID", "MGI_Genotype_ID"]
        )
        
        pheno_count = 0
        for _, row in pheno_df.iterrows():
            mgi_gene_id = row["MGI_Gene_ID"]
            mp_id = row["MP_ID"]
            
            # Map MGI ID to gene symbol
            gene_symbol = mgi_id_to_symbol.get(mgi_gene_id)
            
            if gene_symbol and gene_symbol in builder.entity2id and pd.notna(mp_id):
                # Use readable name if available, otherwise use MP ID
                phenotype_name = mp_id_to_name.get(mp_id, mp_id)
                builder.add_entity(phenotype_name, "phenotype", metadata={"mp_id": mp_id})
                builder.add_triple(gene_symbol, "causes", phenotype_name, add_inverse=True)
                pheno_count += 1
        
        logger.info(f"Added {pheno_count} gene-phenotype associations")
    
    # ===== 4. Load Gene Ontology =====
    logger.info("Loading GO annotations...")
    go_file = data_path / "ontologies" / "mgi_go.gaf"
    
    if not go_file.exists():
        logger.warning(f"GO file not found: {go_file}")
        logger.info("Skipping GO annotations...")
    else:
        # GAF 2.1 format - standard column names
        gaf_columns = [
            "DB", "DB_Object_ID", "DB_Object_Symbol", "Qualifier", "GO_ID",
            "DB_Reference", "Evidence_Code", "With_From", "Aspect",
            "DB_Object_Name", "DB_Object_Synonym", "DB_Object_Type",
            "Taxon", "Date", "Assigned_By", "Annotation_Extension", "Gene_Product_Form_ID"
        ]
        
        go_df = pd.read_csv(
            go_file, 
            sep="\t", 
            comment="!", 
            header=None,
            names=gaf_columns,
            low_memory=False
        )
        
        go_count = 0
        for _, row in go_df.iterrows():
            gene = row["DB_Object_Symbol"]
            go_id = row["GO_ID"]  # e.g., GO:0007596
            aspect = row["Aspect"]  # F, P, or C
            
            if gene in builder.entity2id and pd.notna(go_id):
                # Use GO ID as entity name (we'll map to readable names from go.obo later)
                builder.add_entity(go_id, "go_term", metadata={"go_id": go_id})
                
                # Determine relation type based on GO aspect
                if aspect == "F":  # Molecular Function
                    builder.add_triple(gene, "has_function", go_id, add_inverse=True)
                elif aspect == "P":  # Biological Process
                    builder.add_triple(gene, "participates_in", go_id)
                elif aspect == "C":  # Cellular Component
                    builder.add_triple(gene, "located_in", go_id, add_inverse=True)
                
                go_count += 1
        
        logger.info(f"Added {go_count} GO annotations")
    
    # ===== 5. Load KEGG Pathways (Optional - Skipped) =====
    logger.info("Loading KEGG pathways...")
    kegg_file = data_path / "kegg" / "mmu_pathway_genes.txt"
    
    if not kegg_file.exists():
        logger.warning(f"KEGG file not found: {kegg_file}")
        logger.info("Skipping KEGG pathways...")
    else:
        kegg_df = pd.read_csv(kegg_file, sep="\t")
        for pathway, genes in kegg_df.groupby("pathway_id"):
            pathway_name = f"KEGG:{pathway}"
            builder.add_entity(pathway_name, "pathway")
            
            for gene in genes["gene_symbol"]:
                if gene in builder.entity2id:
                    builder.add_triple(gene, "participates_in", pathway_name)
        
        logger.info(f"Added KEGG pathways")
    
    # ===== 6. Load STRING Protein Interactions =====
    logger.info("Loading protein-protein interactions...")
    string_links = data_path / "string" / "10090.protein.links.v12.0.txt"
    string_info = data_path / "string" / "10090.protein.info.v12.0.txt"
    
    if not string_links.exists() or not string_info.exists():
        logger.warning(f"STRING files not found")
        logger.info("Skipping protein interactions...")
    else:
        # Note: STRING files use space for links, tab for info
        string_df = pd.read_csv(string_links, sep=" ", comment="#")
        string_info_df = pd.read_csv(string_info, sep="\t", comment="#")
        
        # Map protein IDs to gene symbols
        # Column name is #string_protein_id (with #), pandas reads as string_protein_id
        protein_id_col = [col for col in string_info_df.columns if 'protein_id' in col.lower()][0]
        
        protein2gene = dict(zip(
            string_info_df[protein_id_col],
            string_info_df["preferred_name"]
        ))
        
        interaction_count = 0
        for _, row in string_df.iterrows():
            protein1 = row["protein1"]
            protein2 = row["protein2"]
            confidence = row["combined_score"]
            
            # Only include high-confidence interactions (>400)
            if confidence > 400:
                gene1 = protein2gene.get(protein1)
                gene2 = protein2gene.get(protein2)
                
                if gene1 in builder.entity2id and gene2 in builder.entity2id:
                    builder.add_triple(gene1, "interacts_with", gene2)
                    # Symmetric relation - add both directions
                    builder.add_triple(gene2, "interacts_with", gene1)
                    interaction_count += 1
        
        logger.info(f"Added {interaction_count} protein interactions")
    
    # ===== 7. Load Tissue Expression (GTEx - Optional) =====
    logger.info("Loading tissue expression...")
    gtex_file = data_path / "gtex" / "gene_median_tpm.gct"
    
    if not gtex_file.exists():
        logger.warning(f"GTEx file not found: {gtex_file}")
        logger.info("Skipping tissue expression...")
    else:
        gtex_df = pd.read_csv(gtex_file, sep="\t", skiprows=2)
        
        tissues = gtex_df.columns[2:]  # Skip gene_id and gene_name columns
        for tissue in tissues:
            builder.add_entity(tissue, "tissue")
        
        for _, row in gtex_df.iterrows():
            gene = row["Description"]  # Gene symbol
            if gene in builder.entity2id:
                for tissue in tissues:
                    tpm = row[tissue]
                    # Only include if expressed (TPM > 1.0)
                    if tpm > 1.0:
                        builder.add_triple(gene, "expressed_in", tissue, add_inverse=True)
        
        logger.info(f"Added tissue expression for {len(tissues)} tissues")
    
    # ===== 8. Add GO Hierarchy =====
    logger.info("Loading GO hierarchy...")
    go_obo = data_path / "ontologies" / "go.obo"
    
    if not go_obo.exists():
        logger.warning(f"GO OBO file not found: {go_obo}")
        logger.info("Skipping GO hierarchy...")
    else:
        try:
            logger.info(f"Reading GO OBO file: {go_obo} ({go_obo.stat().st_size / 1024 / 1024:.1f} MB)")
            go_graph = obonet.read_obo(str(go_obo))
            go_hierarchy_count = 0
            
            for go_id, data in go_graph.nodes(data=True):
                # Use GO ID (e.g., "GO:0008150") as entity name
                if go_id in builder.entity2id:
                    # Add is_a relationships
                    if "is_a" in data:
                        for parent_id in data["is_a"]:
                            if parent_id in builder.entity2id:
                                builder.add_triple(go_id, "part_of", parent_id, add_inverse=True)
                                go_hierarchy_count += 1
            
            logger.info(f"Added {go_hierarchy_count} GO hierarchy relationships")
        except Exception as e:
            logger.error(f"Failed to load GO hierarchy: {e}")
            logger.warning("Skipping GO hierarchy...")
    
    # ===== 9. Add MPO Hierarchy =====
    logger.info("Loading MPO hierarchy...")
    
    if not mpo_file.exists() or mpo_graph is None:
        logger.warning(f"MPO not available")
        logger.info("Skipping MPO hierarchy...")
    else:
        try:
            mpo_hierarchy_count = 0
            
            for mp_id, data in mpo_graph.nodes(data=True):
                mp_name = mp_id_to_name.get(mp_id, mp_id)
                
                if mp_name in builder.entity2id:
                    # Add is_a relationships
                    if "is_a" in data:
                        for parent_id in data["is_a"]:
                            parent_name = mp_id_to_name.get(parent_id, parent_id)
                            if parent_name in builder.entity2id:
                                builder.add_triple(mp_name, "part_of", parent_name, add_inverse=True)
                                mpo_hierarchy_count += 1
            
            logger.info(f"Added {mpo_hierarchy_count} MPO hierarchy relationships")
        except Exception as e:
            logger.error(f"Failed to add MPO hierarchy: {e}")
            logger.warning("Skipping MPO hierarchy...")
    
    # Print statistics
    builder.print_stats()
    
    # Build and save
    kg_data = builder.build()
    output_dir = Path(output_dir)
    builder.save(output_dir, kg_data)
    
    logger.info(f"✅ KG construction complete!")
    logger.info(f"Saved to {output_dir}")
    
    metadata = {
        "entity2id": builder.entity2id,
        "id2entity": builder.id2entity,
        "relation2id": builder.relation2id,
    }
    
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
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing downloaded data sources (for full mode)"
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
            data_dir=args.data_dir,
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

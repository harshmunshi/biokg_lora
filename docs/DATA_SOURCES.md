# Data Sources Guide

How to download and prepare biological data for BioKG-LoRA.

---

## Overview

BioKG-LoRA integrates data from 6 public biological databases to create a unified knowledge graph.

**Total Expected Size**: ~87,000 entities, ~1,500,000 triples

---

## Data Directory Structure

After downloading, your `data/` directory should look like:

```
data/
├── mgi/
│   ├── MRK_List2.rpt                    # Gene list (~23K genes)
│   └── MGI_PhenoGenoMP.rpt             # Gene-phenotype associations
├── ontologies/
│   ├── mgi_go.gaf                       # GO annotations for mouse
│   └── go.obo                           # GO hierarchy
├── kegg/
│   └── mmu_pathway_genes.txt            # KEGG pathways (requires processing)
├── string/
│   ├── 10090.protein.links.v12.0.txt   # Protein interactions
│   └── 10090.protein.info.v12.0.txt    # Protein metadata
└── gtex/ (optional)
    └── gene_median_tpm.gct              # Tissue expression (human)
```

---

## 1. MGI (Mouse Genome Informatics)

**Purpose**: Mouse genes and phenotypes

### Download Instructions

```bash
# Create directory
mkdir -p data/mgi
cd data/mgi

# Download gene list
wget http://www.informatics.jax.org/downloads/reports/MRK_List2.rpt

# Download gene-phenotype associations
wget http://www.informatics.jax.org/downloads/reports/MGI_PhenoGenoMP.rpt
```

**Files**:
- `MRK_List2.rpt`: ~23,000 mouse genes
- `MGI_PhenoGenoMP.rpt`: ~95,000 gene-phenotype associations

**Format**: Tab-separated values (TSV)

**License**: Free for research use

**Website**: http://www.informatics.jax.org/

---

## 2. Gene Ontology (GO)

**Purpose**: Gene functions and biological processes

### Download Instructions

```bash
# Create directory
mkdir -p data/ontologies
cd data/ontologies

# Download GO annotations for mouse
wget http://current.geneontology.org/annotations/mgi.gaf.gz
gunzip mgi.gaf.gz
mv mgi.gaf mgi_go.gaf

# Download GO ontology (hierarchy)
wget http://purl.obolibrary.org/obo/go.obo
```

**Files**:
- `mgi_go.gaf`: Gene Ontology annotations for mouse genes
- `go.obo`: GO term hierarchy (~46,000 terms)

**Format**: 
- GAF: Tab-separated with headers
- OBO: Ontology format (parsed with `obonet`)

**License**: CC BY 4.0

**Website**: http://geneontology.org/

---

## 3. KEGG (Kyoto Encyclopedia of Genes and Genomes)

**Purpose**: Biological pathways

### Download Instructions

KEGG requires registration and has usage restrictions.

**Option A: KEGG API** (Recommended)

```python
from bioservices import KEGG

k = KEGG()

# Get mouse pathways
pathways = k.list("pathway", "mmu")  # mmu = Mus musculus

# Get genes for each pathway
import pandas as pd

data = []
for pathway_id in pathways:
    genes = k.get(pathway_id)
    # Parse and extract genes
    # ... (see biokg_lora/data/data_sources.py)
    
# Save to file
df = pd.DataFrame(data)
df.to_csv("data/kegg/mmu_pathway_genes.txt", sep="\t", index=False)
```

**Option B: Manual Download**

1. Go to: https://www.genome.jp/kegg/
2. Register for academic license
3. Download mouse pathways
4. Extract gene-pathway mappings

**Expected**: ~300 pathways, ~8,000 genes

**License**: Academic license required

**Website**: https://www.genome.jp/kegg/

---

## 4. STRING (Protein Interactions)

**Purpose**: Protein-protein interactions

### Download Instructions

```bash
# Create directory
mkdir -p data/string
cd data/string

# Download protein links (10090 = Mus musculus)
wget https://stringdb-downloads.org/download/protein.links.v12.0/10090.protein.links.v12.0.txt.gz
gunzip 10090.protein.links.v12.0.txt.gz

# Download protein info (for gene names)
wget https://stringdb-downloads.org/download/protein.info.v12.0/10090.protein.info.v12.0.txt.gz
gunzip 10090.protein.info.v12.0.txt.gz
```

**Files**:
- `10090.protein.links.v12.0.txt`: ~450,000 interactions
- `10090.protein.info.v12.0.txt`: Protein metadata

**Format**: Space-separated values

**Note**: Filter for high-confidence interactions (score > 400)

**License**: CC BY 4.0

**Website**: https://string-db.org/

---

## 5. MPO (Mammalian Phenotype Ontology)

**Purpose**: Phenotype hierarchy

### Download Instructions

```bash
cd data/ontologies

# Download MPO
wget http://www.informatics.jax.org/downloads/reports/MPheno_OBO.ontology
mv MPheno_OBO.ontology mpo.obo
```

**File**: `mpo.obo`: ~12,000 phenotype terms

**Format**: OBO format

**License**: Free for research

**Website**: http://www.informatics.jax.org/

---

## 6. GTEx (Optional - Human Tissue Expression)

**Purpose**: Tissue-specific gene expression

### Download Instructions

GTEx requires registration.

1. Go to: https://gtexportal.org/home/downloads/adult-gtex
2. Register (free for research)
3. Download: `GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct.gz`
4. Gunzip and place in `data/gtex/`

**Note**: 
- GTEx is human data (may need gene ortholog mapping for mouse)
- Alternative: Use mouse expression data from MGI or other sources
- Optional: Can skip if not using tissue expression

**File Size**: ~500 MB (compressed), ~2 GB (uncompressed)

**License**: dbGaP controlled access (or public summary data)

**Website**: https://gtexportal.org/

---

## Quick Download Script

```bash
#!/bin/bash

# Create directories
mkdir -p data/{mgi,ontologies,kegg,string,gtex}

echo "Downloading MGI data..."
cd data/mgi
wget -q http://www.informatics.jax.org/downloads/reports/MRK_List2.rpt
wget -q http://www.informatics.jax.org/downloads/reports/MGI_PhenoGenoMP.rpt

echo "Downloading GO data..."
cd ../ontologies
wget -q http://current.geneontology.org/annotations/mgi.gaf.gz
gunzip -q mgi.gaf.gz
mv mgi.gaf mgi_go.gaf
wget -q http://purl.obolibrary.org/obo/go.obo

echo "Downloading MPO..."
wget -q http://www.informatics.jax.org/downloads/reports/MPheno_OBO.ontology
mv MPheno_OBO.ontology mpo.obo

echo "Downloading STRING data..."
cd ../string
wget -q https://stringdb-downloads.org/download/protein.links.v12.0/10090.protein.links.v12.0.txt.gz
wget -q https://stringdb-downloads.org/download/protein.info.v12.0/10090.protein.info.v12.0.txt.gz
gunzip -q *.gz

echo "✅ Download complete!"
echo ""
echo "Note: KEGG and GTEx require manual download (registration required)"
echo "  - KEGG: https://www.genome.jp/kegg/"
echo "  - GTEx: https://gtexportal.org/"
echo ""
echo "Run: python scripts/stage0_build_kg.py --mode full"
```

Save as `download_data.sh` and run:

```bash
chmod +x download_data.sh
./download_data.sh
```

---

## Data Size Estimates

| Source | Files | Compressed | Uncompressed |
|--------|-------|------------|--------------|
| MGI | 2 files | ~30 MB | ~100 MB |
| GO | 2 files | ~50 MB | ~200 MB |
| STRING | 2 files | ~200 MB | ~800 MB |
| MPO | 1 file | ~5 MB | ~15 MB |
| KEGG | 1 file | ~2 MB | ~5 MB |
| GTEx | 1 file | ~500 MB | ~2 GB |
| **Total** | **9 files** | **~787 MB** | **~3.1 GB** |

---

## Expected KG Statistics

After processing all data sources:

```
Entities: 87,452
  - gene: 23,419 (26.8%)
  - go_term: 45,891 (52.5%)
  - phenotype: 11,854 (13.6%)
  - protein: 5,906 (6.8%)
  - pathway: 328 (0.4%)
  - tissue: 54 (0.1%)

Triples: 1,458,203
  - interacts_with: 450,000 (30.9%)
  - expressed_in: 180,000 (12.3%)
  - part_of: 120,000 (8.2%)
  - participates_in: 120,000 (8.2%)
  - causes: 95,000 (6.5%)
  - has_function: 78,000 (5.3%)
  - (other relations): 415,203 (28.5%)
```

---

## Troubleshooting

### Issue: Download fails

**Solution**: Check internet connection, try manual download from website

### Issue: File format changed

**Solution**: Check data source website for updated format documentation

### Issue: Missing files

**Solution**: Some files are optional (GTEx, KEGG). KG can be built without them.

### Issue: KEGG access denied

**Solution**: 
- Use KEGG API with academic license
- Or skip KEGG (optional)
- Or use alternative pathway database (Reactome, WikiPathways)

---

## Alternative: Use Dummy Data

If you can't download real data, use dummy KG for testing:

```bash
python scripts/stage0_build_kg.py --mode dummy --num_genes 1000
```

This creates a synthetic KG with:
- 1000 genes
- 500 phenotypes
- 2000 GO terms
- 100 pathways
- 50 tissues
- Random edges

---

## Next Steps

After downloading data:

```bash
# Build KG from real data
python scripts/stage0_build_kg.py --mode full --data_dir data

# This will create:
# - data/kg/biological_kg.pt
# - data/kg/entity2id.json
# - data/kg/id2entity.json
# - data/kg/kg_stats.json
```

Then proceed to Stage 1 (RotatE training).

---

## License Compliance

**Important**: Check each data source's license:

- ✅ **MGI**: Free for research, cite in publications
- ✅ **GO**: CC BY 4.0, free to use
- ✅ **STRING**: CC BY 4.0, cite in publications
- ✅ **MPO**: Free for research
- ⚠️ **KEGG**: Academic license required for full access
- ⚠️ **GTEx**: Registration required, controlled access for some data

Always cite the original data sources in publications!

---

## Citations

```bibtex
@article{mgi2021,
  title={Mouse Genome Informatics (MGI)},
  journal={Nucleic Acids Research},
  year={2021}
}

@article{go2021,
  title={The Gene Ontology resource},
  journal={Nucleic Acids Research},
  year={2021}
}

@article{string2021,
  title={STRING v11},
  journal={Nucleic Acids Research},
  year={2021}
}
```

---

**Last Updated**: January 2026  
**Data Versions**: Latest as of January 2026

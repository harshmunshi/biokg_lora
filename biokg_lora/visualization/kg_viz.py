"""
Knowledge Graph Visualization Tools.

Provides:
- Interactive HTML visualization (Pyvis)
- Static graph visualization (Matplotlib/NetworkX)
- Subgraph exploration
- Statistical dashboards
"""

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
import torch
from pyvis.network import Network

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simple Data class (avoids PyG dependency for macOS compatibility)
class Data:
    """Simple data container for knowledge graph"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @property
    def num_nodes(self):
        if hasattr(self, 'x') and self.x is not None:
            return self.x.size(0)
        elif hasattr(self, 'edge_index') and self.edge_index is not None:
            return int(self.edge_index.max()) + 1
        return 0


def visualize_kg_interactive(
    kg_path: str,
    entity2id_path: str,
    output_html: str = "kg_visualization.html",
    max_nodes: int = 500,
    layout: str = "spring",
    height: str = "800px",
    width: str = "100%",
    entity_type_colors: Optional[Dict[str, str]] = None,
):
    """
    Create interactive HTML visualization of the knowledge graph.
    
    Args:
        kg_path: Path to KG PyG Data file
        entity2id_path: Path to entity2id.json
        output_html: Output HTML file path
        max_nodes: Maximum nodes to visualize (for performance)
        layout: Layout algorithm ("spring", "circular", etc.)
        height: Canvas height
        width: Canvas width
        entity_type_colors: Dict mapping entity types to colors
    """
    logger.info(f"Loading KG from {kg_path}...")
    
    # Load KG (weights_only=False needed for PyG Data objects)
    kg_data = torch.load(kg_path, weights_only=False)
    
    # Load entity mappings
    with open(entity2id_path) as f:
        entity2id = json.load(f)
    id2entity = {v: k for k, v in entity2id.items()}
    
    # Sample nodes if too large
    if kg_data.num_nodes > max_nodes:
        logger.warning(f"KG has {kg_data.num_nodes} nodes. Sampling {max_nodes} for visualization.")
        sampled_nodes = np.random.choice(kg_data.num_nodes, max_nodes, replace=False)
        sampled_nodes = set(sampled_nodes.tolist())
    else:
        sampled_nodes = set(range(kg_data.num_nodes))
    
    # Default colors
    if entity_type_colors is None:
        entity_type_colors = {
            0: "#FF6B6B",  # gene - red
            1: "#4ECDC4",  # pathway - teal
            2: "#45B7D1",  # go_term - blue
            3: "#FFA07A",  # phenotype - orange
            4: "#98D8C8",  # tissue - green
            5: "#F7DC6F",  # protein - yellow
        }
    
    # Create Pyvis network
    net = Network(height=height, width=width, directed=True)
    net.barnes_hut()
    
    # Add nodes
    logger.info("Adding nodes...")
    for node_id in sampled_nodes:
        if node_id < kg_data.num_nodes:
            entity_name = id2entity.get(node_id, f"Entity_{node_id}")
            entity_type = kg_data.entity_type[node_id].item() if hasattr(kg_data, 'entity_type') else 0
            color = entity_type_colors.get(entity_type, "#95A5A6")
            
            net.add_node(
                node_id,
                label=entity_name[:20],  # Truncate long names
                title=entity_name,
                color=color,
                size=20,
            )
    
    # Add edges
    logger.info("Adding edges...")
    edge_index = kg_data.edge_index.T
    edge_type = kg_data.edge_type if hasattr(kg_data, 'edge_type') else None
    
    for i in range(edge_index.size(0)):
        src, dst = edge_index[i].tolist()
        
        if src in sampled_nodes and dst in sampled_nodes:
            rel_type = edge_type[i].item() if edge_type is not None else 0
            net.add_edge(src, dst, title=f"Relation {rel_type}")
    
    # Customize physics
    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -8000,
          "centralGravity": 0.3,
          "springLength": 95,
          "springConstant": 0.04
        }
      }
    }
    """)
    
    # Save
    net.save_graph(output_html)
    logger.info(f"Interactive visualization saved to {output_html}")


def visualize_subgraph(
    kg_path: str,
    entity2id_path: str,
    center_entity: str,
    hops: int = 2,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 15),
    layout: str = "spring",
):
    """
    Visualize k-hop subgraph around a center entity.
    
    Args:
        kg_path: Path to KG PyG Data file
        entity2id_path: Path to entity2id.json
        center_entity: Name of center entity
        hops: Number of hops
        output_path: Output image path (None = show)
        figsize: Figure size
        layout: Layout algorithm
    """
    logger.info(f"Extracting {hops}-hop subgraph around '{center_entity}'...")
    
    # Load KG (weights_only=False needed for PyG Data objects)
    kg_data = torch.load(kg_path, weights_only=False)
    
    # Load entity mappings
    with open(entity2id_path) as f:
        entity2id = json.load(f)
    id2entity = {v: k for k, v in entity2id.items()}
    
    # Get center node ID
    if center_entity not in entity2id:
        logger.error(f"Entity '{center_entity}' not found in KG")
        return
    
    center_id = entity2id[center_entity]
    
    # Convert to NetworkX
    G = nx.DiGraph()
    edge_index = kg_data.edge_index.T
    
    for i in range(edge_index.size(0)):
        src, dst = edge_index[i].tolist()
        G.add_edge(src, dst)
    
    # Extract k-hop subgraph
    nodes_in_subgraph = {center_id}
    current_frontier = {center_id}
    
    for _ in range(hops):
        next_frontier = set()
        for node in current_frontier:
            # Add neighbors
            neighbors = set(G.successors(node)) | set(G.predecessors(node))
            next_frontier.update(neighbors)
        
        nodes_in_subgraph.update(next_frontier)
        current_frontier = next_frontier
    
    subgraph = G.subgraph(nodes_in_subgraph).copy()
    
    logger.info(f"Subgraph has {subgraph.number_of_nodes()} nodes, "
               f"{subgraph.number_of_edges()} edges")
    
    # Visualize
    plt.figure(figsize=figsize)
    
    # Layout
    if layout == "spring":
        pos = nx.spring_layout(subgraph, k=0.5, iterations=50)
    elif layout == "circular":
        pos = nx.circular_layout(subgraph)
    else:
        pos = nx.kamada_kawai_layout(subgraph)
    
    # Color by distance from center
    distances = nx.single_source_shortest_path_length(
        subgraph.to_undirected(), center_id, cutoff=hops
    )
    node_colors = [distances.get(node, hops) for node in subgraph.nodes()]
    
    # Draw
    nx.draw_networkx_nodes(
        subgraph, pos,
        node_color=node_colors,
        cmap=plt.cm.RdYlBu_r,
        node_size=500,
        alpha=0.8,
    )
    
    nx.draw_networkx_edges(
        subgraph, pos,
        alpha=0.3,
        arrows=True,
        arrowsize=10,
        edge_color='gray',
    )
    
    # Labels
    labels = {node: id2entity.get(node, f"E{node}")[:15] for node in subgraph.nodes()}
    nx.draw_networkx_labels(
        subgraph, pos,
        labels,
        font_size=8,
    )
    
    # Highlight center
    nx.draw_networkx_nodes(
        subgraph, pos,
        nodelist=[center_id],
        node_color='red',
        node_size=800,
        alpha=1.0,
    )
    
    plt.title(f"{hops}-hop Neighborhood of '{center_entity}'", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Subgraph saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def create_kg_dashboard(
    kg_path: str,
    entity2id_path: str,
    output_html: str = "kg_dashboard.html",
):
    """
    Create a statistical dashboard for the knowledge graph.
    
    Args:
        kg_path: Path to KG PyG Data file
        entity2id_path: Path to entity2id.json
        output_html: Output HTML file path
    """
    logger.info("Creating KG dashboard...")
    
    # Load KG (weights_only=False needed for PyG Data objects)
    kg_data = torch.load(kg_path, weights_only=False)
    
    # Load entity mappings
    with open(entity2id_path) as f:
        entity2id = json.load(f)
    
    # Convert to NetworkX
    G = nx.DiGraph()
    edge_index = kg_data.edge_index.T
    edge_type = kg_data.edge_type if hasattr(kg_data, 'edge_type') else None
    
    for i in range(edge_index.size(0)):
        src, dst = edge_index[i].tolist()
        rel = edge_type[i].item() if edge_type is not None else 0
        G.add_edge(src, dst, relation=rel)
    
    # Compute statistics
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
    in_degrees = [d for n, d in G.in_degree()]
    out_degrees = [d for n, d in G.out_degree()]
    
    # Create figures
    fig = go.Figure()
    
    # Degree distribution
    fig.add_trace(go.Histogram(
        x=degree_sequence,
        name="Degree Distribution",
        opacity=0.7,
    ))
    
    fig.update_layout(
        title="Knowledge Graph Statistics Dashboard",
        xaxis_title="Degree",
        yaxis_title="Count",
        showlegend=True,
    )
    
    # Add summary statistics as annotation
    stats_text = f"""
    <b>Graph Statistics:</b><br>
    Nodes: {G.number_of_nodes():,}<br>
    Edges: {G.number_of_edges():,}<br>
    Avg Degree: {np.mean(degree_sequence):.2f}<br>
    Max Degree: {max(degree_sequence)}<br>
    Density: {nx.density(G):.4f}
    """
    
    fig.add_annotation(
        text=stats_text,
        xref="paper", yref="paper",
        x=0.95, y=0.95,
        showarrow=False,
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
    )
    
    fig.write_html(output_html)
    logger.info(f"Dashboard saved to {output_html}")


if __name__ == "__main__":
    # Test with dummy KG
    from biokg_lora.data.kg_builder import create_dummy_kg
    import tempfile
    
    # Create dummy KG
    kg_data, metadata = create_dummy_kg(num_genes=100, num_phenotypes=50)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Save KG
        kg_path = tmpdir / "test_kg.pt"
        entity2id_path = tmpdir / "entity2id.json"
        
        torch.save(kg_data, kg_path)
        with open(entity2id_path, "w") as f:
            json.dump(metadata["entity2id"], f)
        
        # Test interactive visualization
        visualize_kg_interactive(
            kg_path=str(kg_path),
            entity2id_path=str(entity2id_path),
            output_html="test_kg_interactive.html",
            max_nodes=100,
        )
        
        # Test subgraph visualization
        visualize_subgraph(
            kg_path=str(kg_path),
            entity2id_path=str(entity2id_path),
            center_entity="Gene0000",
            hops=2,
            output_path="test_subgraph.png",
        )
        
        # Test dashboard
        create_kg_dashboard(
            kg_path=str(kg_path),
            entity2id_path=str(entity2id_path),
            output_html="test_dashboard.html",
        )
        
        print("Visualizations created successfully!")

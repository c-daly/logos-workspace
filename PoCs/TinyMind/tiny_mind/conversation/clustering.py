"""Topic clustering for knowledge graph visualization.

Uses Louvain community detection to identify topic clusters,
and provides color generation for visual distinction.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..substrate.node import Node
    from ..substrate.edge import Edge


def detect_communities(nodes: list["Node"], edges: list["Edge"]) -> dict[str, int]:
    """
    Detect topic communities using Louvain algorithm.

    Args:
        nodes: List of Node objects
        edges: List of Edge objects

    Returns:
        Mapping of node_id -> cluster_id (0-indexed).
        Returns empty dict if networkx not available or graph too small.
    """
    try:
        import networkx as nx
    except ImportError:
        return {}

    if len(nodes) < 2:
        return {}

    # Build undirected graph for community detection
    G = nx.Graph()
    for node in nodes:
        G.add_node(node.id)
    for edge in edges:
        # Add edge if both endpoints exist
        if G.has_node(edge.source_id) and G.has_node(edge.target_id):
            G.add_edge(edge.source_id, edge.target_id)

    # Detect communities
    try:
        communities = nx.community.louvain_communities(G, seed=42)
    except Exception:
        return {}

    # Convert to node_id -> cluster_id mapping
    node_to_cluster = {}
    for cluster_id, community in enumerate(communities):
        for node_id in community:
            node_to_cluster[node_id] = cluster_id

    return node_to_cluster


def generate_cluster_colors(num_clusters: int) -> list[str]:
    """
    Generate visually distinct colors for clusters.

    Uses colorblind-friendly palette for small cluster counts,
    falls back to golden angle distribution for larger counts.

    Args:
        num_clusters: Number of clusters to generate colors for

    Returns:
        List of hex color strings (e.g., ["#ff6b6b", "#4ecdc4", ...])
    """
    if num_clusters == 0:
        return []

    # High-contrast colorblind-friendly palette
    # Maximizes differences in hue, saturation, AND lightness
    colorblind_safe = [
        "#E31A1C",  # Bright red
        "#1F78B4",  # Strong blue
        "#33A02C",  # Bright green
        "#FF7F00",  # Bright orange
        "#6A3D9A",  # Purple
        "#FFFF33",  # Bright yellow
        "#A6CEE3",  # Light blue
        "#FB9A99",  # Light pink/salmon
        "#B2DF8A",  # Light green
        "#CAB2D6",  # Light purple
        "#FDBF6F",  # Light orange
        "#B15928",  # Brown
    ]

    if num_clusters <= len(colorblind_safe):
        return colorblind_safe[:num_clusters]

    # For more clusters, use golden angle but with more separation
    colors = list(colorblind_safe)  # Start with safe colors
    golden_angle = 137.5

    for i in range(len(colorblind_safe), num_clusters):
        hue = (i * golden_angle) % 360
        saturation = 70
        lightness = 60
        hex_color = hsl_to_hex(hue, saturation, lightness)
        colors.append(hex_color)

    return colors


def hsl_to_hex(h: float, s: float, l: float) -> str:
    """Convert HSL (hue 0-360, sat 0-100, light 0-100) to hex color."""
    s = s / 100
    l = l / 100

    c = (1 - abs(2 * l - 1)) * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = l - c / 2

    if h < 60:
        r, g, b = c, x, 0
    elif h < 120:
        r, g, b = x, c, 0
    elif h < 180:
        r, g, b = 0, c, x
    elif h < 240:
        r, g, b = 0, x, c
    elif h < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x

    r = int((r + m) * 255)
    g = int((g + m) * 255)
    b = int((b + m) * 255)

    return f"#{r:02x}{g:02x}{b:02x}"


def get_confidence_border(confidence: float) -> dict:
    """
    Get border styling based on confidence level.

    Args:
        confidence: Confidence value 0-1

    Returns:
        Dict with 'width' and 'color' for pyvis border styling
    """
    if confidence >= 0.7:
        return {"width": 4, "color": "#4ecca3"}  # Green, thick
    elif confidence >= 0.4:
        return {"width": 2, "color": "#ffd93d"}  # Yellow, medium
    else:
        return {"width": 1, "color": "#ff6b6b"}  # Red, thin

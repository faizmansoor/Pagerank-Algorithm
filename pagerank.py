import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def create_transition_matrix(graph):
    """
    Create the transition probability matrix from a directed graph
    """
    n = len(graph)  # Get number of pages
    M = np.zeros((n, n))  # Create empty nxn matrix filled with zeros
    for i in range(n):  # For each page i
        total_links = sum(graph[i])  # Count how many outgoing links page i has
        if total_links == 0:  # If page i has no outgoing links
            # Make it link to every page equally (including itself)
            M[i] = np.ones(n) / n
        else:
            # Distribute probability equally among outgoing links
            M[i] = graph[i] / total_links
    return M

def pagerank(graph, damping=0.85, epsilon=1e-8, max_iterations=100):
    """
    Calculate PageRank values for all nodes
    """
    n = len(graph)  # Number of pages
    # Get transition matrix from our helper function
    M = create_transition_matrix(graph)
    # Initialize PageRank values - everyone starts equal
    pagerank = np.ones(n) / n
    # Create teleportation vector - equal probability to jump anywhere
    v = np.ones(n) / n
    # Power iteration loop
    for iteration in range(max_iterations):  # Fixed the syntax error here
        # Save current values to check for convergence
        prev_pagerank = pagerank.copy()
        # The key PageRank equation:
        # New PR = (damping × transition matrix × old PR) + ((1-damping) × teleportation)
        pagerank = damping * (M.T @ pagerank) + (1 - damping) * v
        # Check if values have stabilized
        if np.sum(np.abs(pagerank - prev_pagerank)) < epsilon:
            print(f"Converged after {iteration+1} iterations")
            break
    return pagerank

def visualize_pagerank(graph, pagerank_values, filename=None):
    """
    Create an enhanced visualization of a graph with PageRank values
    """
    G = nx.from_numpy_array(graph, create_using=nx.DiGraph)
    
    # Calculate node sizes based on PageRank values
    # Scale PageRank values to reasonable node sizes (between 1000 and 3000)
    min_size, max_size = 1000, 3000
    sizes = [min_size + (max_size - min_size) * pr / max(pagerank_values) for pr in pagerank_values]
    
    # Calculate node colors based on PageRank values (from light to dark blue)
    colors = [(0.5 - 0.5 * (pr / max(pagerank_values)), 
               0.5 - 0.3 * (pr / max(pagerank_values)), 
               0.9 - 0.2 * (pr / max(pagerank_values))) for pr in pagerank_values]
    
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, seed=42)  # Fixed seed for reproducibility
    
    # Draw nodes with scaled sizes and colors
    nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color=colors, alpha=0.8)
    
    # Draw edges with varying widths
    edge_weights = [1.5 + 2 * pagerank_values[u] for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.7, arrowsize=20, 
                          arrowstyle='->', connectionstyle='arc3,rad=0.1')
    
    # Add node labels (page numbers)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    
    # Add PageRank values as separate labels
    labels = {i: f'PR: {pr:.3f}' for i, pr in enumerate(pagerank_values)}
    pos_attrs = {node: (coord[0], coord[1] + 0.1) for node, coord in pos.items()}
    nx.draw_networkx_labels(G, pos_attrs, labels, font_size=10, font_color='darkred')
    
    plt.title("Web Graph with PageRank Values", fontsize=16)
    plt.axis('off')
    
    # Add a legend explaining the visualization
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=(0.5, 0.5, 0.9), 
                  markersize=10, label='Low PageRank'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=(0, 0.2, 0.7), 
                  markersize=15, label='High PageRank')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()

# Example usage section:
if __name__ == "__main__":
    # Create example network: 4 pages with these links:
    # Page 0 → Pages 1,2
    # Page 1 → Page 2
    # Page 2 → Pages 0,3
    # Page 3 → Page 2
    sample_graph = np.array([
        [0, 1, 1, 0],  # Row 0: Page 0's outgoing links
        [0, 0, 1, 0],  # Row 1: Page 1's outgoing links
        [1, 0, 0, 1],  # Row 2: Page 2's outgoing links
        [0, 0, 1, 0]   # Row 3: Page 3's outgoing links
    ])
    
    # Calculate PageRank values
    result = pagerank(sample_graph)
    
    # Use enhanced visualization
    visualize_pagerank(sample_graph, result)
    
    # Print final values
    print("\nPageRank values:")
    for i, value in enumerate(result):
        print(f"Page {i}: {value:.3f}")
    
    # Create a larger example for demonstration
    print("\nRunning a larger example (8 nodes):")
    larger_graph = np.zeros((8, 8))
    # Create a more complex link structure
    larger_graph[0, [1, 2, 3]] = 1  # Page 0 links to 1,2,3
    larger_graph[1, [2, 4]] = 1     # Page 1 links to 2,4
    larger_graph[2, [0, 3, 5]] = 1  # Page 2 links to 0,3,5
    larger_graph[3, [4, 6]] = 1     # Page 3 links to 4,6
    larger_graph[4, [5, 7]] = 1     # Page 4 links to 5,7
    larger_graph[5, [2, 6]] = 1     # Page 5 links to 2,6
    larger_graph[6, [3, 7]] = 1     # Page 6 links to 3,7
    larger_graph[7, [0]] = 1        # Page 7 links to 0
    
    # Calculate and visualize
    larger_result = pagerank(larger_graph)
    visualize_pagerank(larger_graph, larger_result)
    
    # Print larger example values
    print("\nLarger graph PageRank values:")
    for i, value in enumerate(larger_result):
        print(f"Page {i}: {value:.3f}")
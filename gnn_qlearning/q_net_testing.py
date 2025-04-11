import torch
import numpy as np
import networkx as nx
from q_gnn import QNetwork
import torch_geometric.utils as pyg_utils
from q_gnn import GraphColoringEnv
from q_gnn import GNN
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time

from mpl_toolkits.mplot3d import Axes3D


def visualize_colored_graph_3d(graph, color_assignments):
    """
    graph: networkx.Graph
    color_assignments: dict {node: color_id}
    """

    # Generate a 3D layout for the nodes
    pos_3d = {node: np.random.rand(3) for node in graph.nodes()}  # Random 3D positions
    cmap = cm.get_cmap('tab20')  # You can try 'tab10', 'Set3', etc.
    node_colors = [cmap(color_assignments[node] % 20) for node in graph.nodes()]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Draw edges
    for u, v in graph.edges():
        x = [pos_3d[u][0], pos_3d[v][0]]
        y = [pos_3d[u][1], pos_3d[v][1]]
        z = [pos_3d[u][2], pos_3d[v][2]]
        ax.plot(x, y, z, color='gray', alpha=0.5)

    # Draw nodes
    for node in graph.nodes():
        x, y, z = pos_3d[node]
        ax.scatter(x, y, z, s=300, color=node_colors[node], edgecolors='black')
        ax.text(x, y, z + 0.03, str(node), fontsize=10, ha='center', va='center')

    ax.set_title("3D Graph Coloring", fontsize=15)
    ax.axis('off')
    plt.show()


def evaluate_agent(env, q_net, edge_index, render=True):
    state = env.reset()
    done = False
    total_reward = 0
    invalid_moves = 0
    used_colors = set()
    node_color_map = dict()

    while not done:
        current_node = env.current_node
        legal_actions = env.get_legal_actions()

        with torch.no_grad():
            q_values = q_net(state, edge_index, current_node)
            mask = torch.tensor([i in legal_actions for i in range(q_values.shape[0])])
            q_values[~mask] = -float('inf')
            action = torch.argmax(q_values).item()

        if action not in legal_actions:
            invalid_moves += 1

        state, reward, done, _ = env.step(action)
        total_reward += reward

        node_color_map[current_node] = action
        if reward == -1:  # new color used
            used_colors.add(action)

    print("Evaluation Result:")
    print(f"Colors used: {len(set(env.colors))}")
    print(f"Invalid moves: {invalid_moves}")
    return node_color_map


def visualize_colored_graph(graph, color_assignments):
    """
    graph: Networkx graph.
    color_assingments: dict {node: color_id}
    """
    # Create color map.
    cmap = cm.get_cmap('tab20')  # 20 distinct colors
    node_colors = [cmap(color_assignments[node] % 20) for node in graph.nodes()]

    pos = nx.spring_layout(graph, seed=42)
    plt.figure(figsize=(8, 6))
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_color=node_colors,
        node_size=600,
        font_weight='bold',
        edge_color='gray'
    )

    plt.savefig("Coloring.png")


def main():
    # Path to testing file.
    path = "mid/DSJC125.9.txt"
    with open(path, "r") as fp:
        lines = fp.read().splitlines()

    # Build adjacency matrix.
    header = lines[0].split(" ")
    nodes = int(header[2])
    edges = int(header[3])
    matrix = np.zeros((nodes, nodes))

    for i in range(edges):
        print(lines[i])
        edge = lines[i + 1].split(" ")
        x = int(edge[1])
        y = int(edge[2])
        matrix[x - 1, y - 1] = 1

    # Get equivalent networkx graph.
    graph = nx.from_numpy_array(matrix)
    MAX_COLORS = 50
    print("\n Testing on a new unseen graph...")
    test_env = GraphColoringEnv(graph, MAX_COLORS)
    test_edge_index = pyg_utils.from_networkx(graph).edge_index

    gnn = GNN(input_dim=MAX_COLORS + 1, hidden_dim=25)
    q_net = QNetwork(gnn, hidden_dim=25, max_colors=MAX_COLORS)
    q_net.load_state_dict(torch.load("q_net_50.pth"))
    q_net.eval()

    start_time = time.time()
    node_color_map = evaluate_agent(test_env, q_net,
                                    test_edge_index, render=True)
    print(f"Testing time: {time.time() - start_time}")
    visualize_colored_graph(graph, node_color_map)
    visualize_colored_graph_3d(graph, node_color_map)


if __name__ == "__main__":
    main()

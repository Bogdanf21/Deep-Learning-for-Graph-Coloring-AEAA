import networkx as nx
import torch
import numpy as np
import random

from torch_geometric.nn import GCNConv
import torch_geometric.utils as pyg_utils

from collections import deque


class GraphColoringEnv:
    def __init__(self, graph: nx.Graph, max_colors: int):
        self.graph = graph
        self.n_nodes = graph.number_of_nodes()
        self.max_colors = max_colors

    def reset(self):
        self.colors = [-1] * self.n_nodes  # -1 = uncolored
        self.current_node = 0
        return self.get_state()

    def get_state(self):
        # One-hot color encoding per node.
        state = np.zeros((self.n_nodes, self.max_colors + 1))
        for i, c in enumerate(self.colors):
            if c == -1:
                state[i, -1] = 1
            else:
                state[i, c] = 1
        return torch.tensor(state, dtype=torch.float32)

    def get_legal_actions(self):
        # Returns list of legal colors for current node.
        illegal = {self.colors[n]
                   for n in self.graph.neighbors(self.current_node)}
        return [c for c in range(self.max_colors) if c not in illegal]

    def step(self, action):  # action = color
        reward = 0
        done = False
        legal_actions = self.get_legal_actions()

        if action not in legal_actions:
            reward = -10  # penalty for illegal color
            done = True
            return self.get_state(), reward, done, {}

        self.colors[self.current_node] = action
        if action not in self.colors[:self.current_node]:
            reward = -4  # penalize new color usage

        self.current_node += 1
        if self.current_node >= self.n_nodes:
            done = True
        return self.get_state(), reward, done, {}


class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GNN, self).__init__()
        print(input_dim, hidden_dim)
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        return x  # embeddings for each node


class QNetwork(torch.nn.Module):
    def __init__(self, gnn, hidden_dim, max_colors):
        super(QNetwork, self).__init__()
        self.gnn = gnn
        self.fc = torch.nn.Linear(hidden_dim, max_colors)

    def forward(self, x, edge_index, current_node):
        embeddings = self.gnn(x, edge_index)
        return self.fc(embeddings[current_node])


def train_dqn(env, q_net, target_net, edge_index, episodes=500, gamma=0.9,
              epsilon=1.0, epsilon_decay=0.995):
    optimizer = torch.optim.Adam(q_net.parameters(), lr=1e-3)
    memory = deque(maxlen=10000)
    batch_size = 32

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            current_node = env.current_node
            legal_actions = env.get_legal_actions()
            if random.random() < epsilon:
                if legal_actions:
                    action = random.choice(legal_actions)
            else:
                with torch.no_grad():
                    q_values = q_net(state, edge_index, current_node)
                    q_values[~torch.tensor(
                        [i in legal_actions for i in range(q_values.shape[0])]
                        )] = -float('inf')
                    action = torch.argmax(q_values).item()

            next_state, reward, done, _ = env.step(action)
            memory.append((state, current_node, action, reward, next_state,
                           done, edge_index))
            state = next_state
            total_reward += reward

            # Train
            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)
                loss = 0

                for s, n, a, r, s_next, d, e in batch:
                    q_pred = q_net(s, e, n)[a]
                    with torch.no_grad():
                        q_next = target_net(s_next, e, n)
                        q_target = r + (0 if d else gamma * torch.max(q_next))
                    loss += (q_pred - q_target) ** 2

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if episode % 10 == 0:
            target_net.load_state_dict(q_net.state_dict())
            print(f"Episode {episode} - Reward: {total_reward:.2f} - Epsilon: {epsilon:.3f}")

        epsilon = max(0.05, epsilon * epsilon_decay)


def evaluate_agent(env, q_net, edge_index, render=True):
    state = env.reset()
    done = False
    total_reward = 0
    invalid_moves = 0
    used_colors = set()

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

        if reward == -1:  # new color used
            used_colors.add(action)

    print(f"Evaluation Result:")
    print(f"Colors used: {len(set(env.colors))}")
    print(f"Invalid moves: {invalid_moves}")


def main():
    MAX_COLORS = 50
    NODES = 100
    # Graph coloring environment.
    G = nx.erdos_renyi_graph(n=NODES, p=0.3, seed=42)
    env = GraphColoringEnv(G, max_colors=MAX_COLORS)

    # Get edge index for Pytorch Geometric.
    edge_index = pyg_utils.from_networkx(G).edge_index

    # Models.
    gnn = GNN(input_dim=MAX_COLORS + 1, hidden_dim=25)
    q_net = QNetwork(gnn, hidden_dim=25, max_colors=MAX_COLORS)
    target_net = QNetwork(GNN(MAX_COLORS+1, 25), hidden_dim=25,
                          max_colors=MAX_COLORS)
    target_net.load_state_dict(q_net.state_dict())

    EPISODES = 1000
    train_dqn(env, q_net, target_net, edge_index, episodes=EPISODES)

    torch.save(q_net.state_dict(), "q_net_50.pth")
    print("Q-network saved to q_net.pth!")

    # === Test on Unseen Graph ===
    # print("\n Testing on a new unseen graph...")
    # test_graph = nx.erdos_renyi_graph(n=NODES, p=0.5, seed=99)
    # test_env = GraphColoringEnv(test_graph, MAX_COLORS)
    # test_edge_index = pyg_utils.from_networkx(test_graph).edge_index

    # evaluate_agent(test_env, q_net, test_edge_index, render=True)


if __name__ == "__main__":
    main()

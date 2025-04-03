import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader
import networkx as nx
import random
from tensorboardX import SummaryWriter
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# Set up device.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def main():
    # Initialize tensorboardX.
    writer = SummaryWriter()

    # Set up hyperparameters.
    num_graphs = 5000
    num_nodes = 10
    edge_prob = 0.2
    k = 3
    hidden_dim = 16
    output_dim = 1
    batch_size = 10
    lr = 0.01
    epochs = 50

    # Prepare dataset and loaders.
    dataset = create_dataset(num_graphs, num_nodes, edge_prob, k)
    train_loader = DataLoader(dataset[:4000], batch_size=batch_size)
    test_loader = DataLoader(dataset[4000:], batch_size=batch_size)

    # Model training.
    model = GCN(input_dim=num_nodes,
                hidden_dim=hidden_dim,
                output_dim=batch_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCELoss()
    train_model(model, train_loader, optimizer, criterion, epochs, writer)
    evaluate_model(model, test_loader, writer)

    # Close TensorBoard writer
    writer.close()


def create_dataset(num_graphs: int, num_nodes: int, edge_prob: float, k: int):
    return [generate_graph(num_nodes, edge_prob, k) for _ in range(num_graphs)]


def generate_graph(num_nodes: int, edge_prob: float, k: int):
    """Generates a random graph and determines if it's k-colorable."""
    G = nx.erdos_renyi_graph(num_nodes, edge_prob)
    try:
        coloring = nx.coloring.greedy_color(G, strategy='largest_first')
        num_colors = len(set(coloring.values()))
        label = 1 if num_colors <= k else 0  # 1 if k-colorable, else 0
    except nx.NetworkXErro:
        label = 0

    edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
    x = torch.eye(num_nodes, dtype=torch.float)
    return Data(x=x.to(device), edge_index=edge_index.to(device),
                y=torch.tensor([label],
                dtype=torch.float).to(device))


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)  # First GCN layer
        self.conv2 = GCNConv(hidden_dim, hidden_dim)  # Second GCN layer
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = torch.mean(x, dim=0)
        x = self.fc(x)
        return torch.sigmoid(x)


def train_model(model, train_loader, optimizer, criterion, epochs, writer):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data in train_loader:
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        writer.add_scalar('Loss/train', total_loss / len(train_loader), epoch)
        print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}')


def evaluate_model(model, test_loader, writer):
    model.eval()
    correct = 0
    total = 0
    embeddings = []
    labels = []
    with torch.no_grad():
        for data in test_loader:
            out = model(data).squeeze()
            pred = (out > 0.5).float()
            correct += (pred == data.y).sum().item()
            total += data.y.size(0)
            embeddings.append(data.x.mean(dim=0).cpu().numpy())
            # labels.append(data.y.item())
    accuracy = correct / total
    writer.add_scalar('Accuracy/test', accuracy)
    print(f'Accuracy: {accuracy:.4f}')
    # visualize_embeddings(embeddings, labels, writer)


def visualize_embeddings(embeddings, labels, writer):
    """Visualizes embeddings using t-SNE."""
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels,
                          cmap='coolwarm', alpha=0.7)
    plt.colorbar(scatter)
    plt.title("t-SNE Visualization of Graph Colorability")
    plt.show()
    writer.add_figure('Embeddings/t-SNE', plt.gcf())


if __name__ == "__main__":
    main()

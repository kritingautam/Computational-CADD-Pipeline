import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool


def initialize_pipeline():
    print("[INFO] Initializing Computational Chemistry GNN Pipeline...")
    print("[INFO] Programmatically fetching NIH Tox21 Molecular Dataset...")

    # Ingest the standard federal screening benchmark via the MoleculeNet wrapper
    dataset = MoleculeNet(root='./data/Tox21', name='Tox21')

    # Programmatically standardize data tensor formats to guarantee uniform batch alignment
    cleaned_dataset = []
    for data in dataset:
        if data.x is not None:
            data.x = data.x.to(torch.float32)  # Force 9-dimensional atom node features to floats
        if data.y is not None:
            data.y = data.y.to(torch.float32)  # Cast multi-label targets to floats to preserve 'nan' masks
        cleaned_dataset.append(data)

    # 80/20 train/test structural subset partitioning
    train_dataset = cleaned_dataset[:6000]
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Define the 3-Layer Graph Convolutional Network Architecture
    class MolecularGNN(torch.nn.Module):
        def __init__(self, hidden_channels, num_node_features, num_classes):
            super(MolecularGNN, self).__init__()
            # Message-passing convolutional configurations mapping covalent bonds
            self.conv1 = GCNConv(num_node_features, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
            self.conv3 = GCNConv(hidden_channels, hidden_channels)
            self.lin = Linear(hidden_channels, num_classes)

        def forward(self, x, edge_index, batch):
            x = self.conv1(x, edge_index).relu()
            x = self.conv2(x, edge_index).relu()
            x = self.conv3(x, edge_index)
            # Global readout pooling to condense atomic features into a single molecular vector
            x = global_mean_pool(x, batch)
            x = F.dropout(x, p=0.5, training=self.training)
            return self.lin(x)

    model = MolecularGNN(hidden_channels=64, num_node_features=dataset.num_node_features,
                         num_classes=dataset.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()

    print(f"[SUCCESS] Dataset compiled! {len(dataset)} molecular graphs loaded.")
    print("Beginning structural training loop simulation...")

    model.train()
    for epoch in range(1, 4):
        total_loss = 0.0  # Standardized to float to eliminate casting exceptions
        for batch in train_loader:
            # Create a boolean mask tracking unrecorded (nan) assays across the 12 columns
            is_labeled = ~torch.isnan(batch.y)

            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)

            # Explicitly force aligned float types across filtered targets and predictions
            predictions = out[is_labeled].to(torch.float32)
            targets = batch.y[is_labeled].to(torch.float32)

            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs

        print(f"Epoch {epoch:02d} | Average Training Loss: {total_loss / len(train_dataset):.4f}")

    print("[SUCCESS] Operational weight parameters successfully optimized.")
    return True


if __name__ == "__main__":
    initialize_pipeline()

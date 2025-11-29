import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml


class QuantumPolicyHead(nn.Module):
    """
    Variational quantum circuit (VQC) wrapped as a PyTorch module.

    Input:  [B*, hidden_dim]
    Output: [B*, n_actions] (policy logits)
    """

    def __init__(
        self,
        hidden_dim: int,
        n_actions: int,
        n_qubits: int = 4,
        n_layers: int = 2,
        q_device: str = "default.qubit",
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_actions = n_actions
        self.n_qubits = n_qubits

        # Compress hidden state to n_qubits features
        self.embed = nn.Linear(hidden_dim, n_qubits)

        # 1) Create device
        dev = qml.device(q_device, wires=n_qubits)

        # 2) Define the circuit and wrap it as a QNode with interface="torch"
        def circuit(inputs, weights):
            # inputs: [n_qubits]
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        weight_shapes = {"weights": (n_layers, n_qubits, 3)}

        qnode = qml.QNode(circuit, dev, interface="torch")

        # 3) Build TorchLayer from the QNode (NO device kwarg, NO init_method)
        self.q_layer = qml.qnn.TorchLayer(
            qnode,
            weight_shapes,
        )

        # Map quantum features -> action logits
        self.out = nn.Linear(n_qubits, n_actions)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: [B*, hidden_dim] (e.g., flattened [batch * time * n_agents, hidden_dim])
        """
        # Project to qubit inputs
        x = self.embed(h)
        x = torch.tanh(x)

        # Quantum layer: returns [B*, n_qubits]
        q_out = self.q_layer(x)

        # Classical readout
        logits = self.out(q_out)  # [B*, n_actions]
        return logits

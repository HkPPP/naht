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

        # PennyLane device
        dev = qml.device(q_device, wires=n_qubits)

        # Define variational circuit
        def circuit(inputs, weights):
            # inputs: shape [n_qubits]
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            # Expectation values as classical outputs
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        # Trainable parameter shapes for the VQC
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}

        # Wrap circuit as a Torch module
        self.q_layer = qml.qnn.TorchLayer(
            circuit,
            weight_shapes,
            device=dev,  # <-- no init_method
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

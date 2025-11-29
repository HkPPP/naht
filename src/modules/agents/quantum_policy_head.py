import torch
import torch.nn as nn
import torch.nn.functional as F

import pennylane as qml


class QuantumPolicyHead(nn.Module):
    """
    Small variational quantum circuit wrapped as a PyTorch module.

    Input:  [B, hidden_dim]  (output of RNN + agent-modeling encoder)
    Output: [B, n_actions]   (policy logits)
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

        # Compress hidden state to n_qubits real values
        self.embed = nn.Linear(hidden_dim, n_qubits)

        # Define quantum device
        dev = qml.device(q_device, wires=n_qubits)

        # Define variational circuit
        def circuit(inputs, weights):
            # inputs: shape [n_qubits]
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            # Expectation values as classical features
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        weight_shapes = {"weights": (n_layers, n_qubits, 3)}

        # Wrap as a TorchLayer
        self.q_layer = qml.qnn.TorchLayer(
            circuit,
            weight_shapes,
            init_method=qml.init.strong_ent_layers_uniform,
            device=dev,
        )

        # Map quantum outputs -> action logits
        self.out = nn.Linear(n_qubits, n_actions)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: [B, hidden_dim] (or [B*T, hidden_dim] after Mac flattening)
        """
        # Project to qubit-sized vector
        x = self.embed(h)
        x = torch.tanh(x)

        # Quantum layer expects shape [..., n_qubits]
        q_out = self.q_layer(x)
        # q_out: [..., n_qubits] of expectation values in [-1, 1]

        logits = self.out(q_out)
        return logits

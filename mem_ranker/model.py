import torch
import torch.nn as nn


class MemoryRanker(nn.Module):

    def __init__(
        self,
        input_dim=384,
        hidden_dim=256,
        num_heads=4,
        num_layers=2,
        max_length=512,
    ):
        super().__init__()

        # Project full clause representations
        self.projection = nn.Linear(input_dim, hidden_dim)
        self.position_embedding = nn.Embedding(max_length, hidden_dim)

        # Encoder layer for clause-level self-attention
        attention_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            activation="gelu",
            dropout=0.1,
            batch_first=True
        )

        # Clause-level self-attention
        self.clause_attention = nn.TransformerEncoder(
            attention_layer,
            num_layers=num_layers
        )

        # Memorability scoring head
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            # Raw logits: pairwise ranking should not be constrained to a
            # bounded [0, 1] difference. Apply sigmoid only for calibration
            # at inference time or in the regression component of training.
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, clause_embeddings, padding_mask=None):

        # [Batch, Clauses, 1536]
        x = self.projection(clause_embeddings)

        # Add a learned representation of each clause's position in the story.
        sequence_length = clause_embeddings.size(1)
        if sequence_length > self.position_embedding.num_embeddings:
            raise ValueError(
                f"Story has {sequence_length} clauses, but max_length is "
                f"{self.position_embedding.num_embeddings}."
            )

        positions = torch.arange(
            sequence_length,
            device=clause_embeddings.device,
        ).unsqueeze(0)
        x = x + self.position_embedding(positions)

        # [Batch, Clauses, Hidden]
        x = self.clause_attention(x, src_key_padding_mask=padding_mask)

        # [Batch, Clauses]
        scores = self.mlp(x).squeeze(-1)

        return scores

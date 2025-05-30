import torch.nn as nn


class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=256,
        dropout=0.1
    ):
        super().__init__()

        """
        Here you should write simple 2-layer MLP consisting:
        2 Linear layers, GELU activation, Dropout and LayerNorm. 
        Do not forget to send a skip-connection right after projection and before LayerNorm.
        The whole structure should be in the following order:
        [Linear, GELU, Linear, Dropout, Skip, LayerNorm]
        """
        self.proj1 = nn.Linear(embedding_dim, projection_dim)
        self.proj2 = nn.Sequential(
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim),
            nn.Dropout(dropout),
        )
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        """
        Perform forward pass, do not forget about skip-connections.
        """
        proj = self.proj1(x)
        proj2 = self.proj2(proj)
        out = proj2 + proj
        out = self.layer_norm(out)
        return out

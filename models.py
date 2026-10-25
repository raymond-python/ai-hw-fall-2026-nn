"""Three small image classifiers for MNIST (28x28 grayscale, 10 classes).

    MLP          - shallow multi-layer perceptron (one hidden layer)
    CNN          - small convolutional network
    Transformer  - tiny Vision-Transformer-style encoder over 7x7 patches
"""
import torch
import torch.nn as nn


class MLP(nn.Module):
    """Shallow MLP: 784 -> 256 -> 10. The image is flattened, so the
    network has no built-in notion of which pixels are neighbours."""

    def __init__(self, hidden=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 10),
        )

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    """Two conv blocks (conv-BN-ReLU x2 + max-pool) and a small classifier.
    28x28 -> 14x14 -> 7x7 feature maps."""

    def __init__(self, c1=32, c2=64, dropout=0.3):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(1, c1), block(c1, c2))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c2 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class TransformerEncoderNet(nn.Module):
    """Tiny ViT: split the image into 4x4 = 16 patches of 7x7 pixels,
    embed each patch, add a [CLS] token and learned position embeddings,
    run a stack of Transformer encoder layers, classify from [CLS]."""

    def __init__(self, patch=7, dim=64, depth=4, heads=4, mlp_dim=128, dropout=0.1):
        super().__init__()
        assert 28 % patch == 0
        self.n_patches = (28 // patch) ** 2
        # A strided conv is the standard way to cut + linearly embed patches.
        self.patch_embed = nn.Conv2d(1, dim, kernel_size=patch, stride=patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=mlp_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 10)

    def forward(self, x):
        x = self.patch_embed(x)                    # (B, dim, 4, 4)
        x = x.flatten(2).transpose(1, 2)           # (B, 16, dim)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        x = self.encoder(x)
        return self.head(self.norm(x[:, 0]))


MODELS = {
    "mlp": MLP,
    "cnn": CNN,
    "transformer": TransformerEncoderNet,
}


def build_model(name):
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(MODELS)}")
    return MODELS[name]()


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

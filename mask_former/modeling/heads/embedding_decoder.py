from detectron2.layers import Conv2d, get_norm
from torch import nn
from torch.nn import functional as F

class EmbeddingDecoder(nn.Module):
    def __init__(self, hidden_dim, num_layers=1, norm=None,):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embed_layer = []
        use_bias = norm == ""
        for _ in range(self.num_layers-1):
            output_norm = get_norm(norm, hidden_dim)
            self.embed_layer.append(
                Conv2d(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=3,
                    padding=1,
                    bias=use_bias,
                    norm=output_norm,
                    activation=F.relu,
                )
            )
        self.embed_layer.append(Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1))
        self.embed_layer = nn.ModuleList(self.embed_layer)
    def forward(self, x):
        for i in range(self.num_layers):
            x = self.embed_layer[i](x)
        return x #b,c,h,w
    



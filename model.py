import torch
import timm

class Backbone(torch.nn.Module):
    def __init__(
        self,
        model_name: str,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward_features(x)

class CTran(torch.nn.Module):
    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        embedding_dim: int = 2048,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        **factory_kwargs
    ) -> None:
        super().__init__()
        self.backbone = Backbone(backbone_name)
        self.num_classes = num_classes
        self.state_embedding = torch.nn.Parameter(torch.rand(3, embedding_dim))
        self.label_embedding = torch.nn.Parameter(torch.rand(1, num_classes, embedding_dim))
        self.layernorm = torch.nn.LayerNorm(embedding_dim)
        encoder_layer = torch.nn.TransformerEncoderLayer(embedding_dim, nhead, embedding_dim, batch_first=True,
                                                    **factory_kwargs)
        encoder_norm = torch.nn.LayerNorm(embedding_dim, **factory_kwargs)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self.tower = torch.nn.Linear(embedding_dim, self.num_classes)

    def forward(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x [B, C, H, W]
        # y [B, L]
        # m [B, L]
        
        assert y.shape == mask.shape, "y and mask should have the same shape"

        # prepare label embedding
        label_embedding = self.label_embedding.repeat(x.size(0), 1, 1)
        y = torch.where(mask == 0, y, 2).long()
        state_embedding = self.state_embedding[y]
        extra_embedding = label_embedding + state_embedding
        
        assert extra_embedding.size(-2) == self.num_classes, "num_class given in mask and y does not match with initialization"

        # get extracted features
        x = self.backbone(x)
        
        # flatten h & w
        x = x.flatten(start_dim=-2, end_dim=-1)
        x = x.permute(0, 2, 1)

        x = torch.concat((x, extra_embedding), dim=-2)
        x = self.layernorm(x)
        x = self.encoder(x)
        x = x[:, -self.num_classes:, :]
        x = self.tower(x)
        diag_mask = torch.eye(self.num_classes).unsqueeze(0).repeat(x.size(0),1,1).to(x.device)
        x = (x*diag_mask).sum(-1)
        return x.squeeze()

if __name__ == "__main__":
    net = CTran("resnet50", 10)
    x = torch.rand(2, 3, 224, 224)
    y = torch.randint(0, 2, (2, 10))
    m = torch.randint(0, 2, (2, 10))
    print(net(x,y,m).shape)
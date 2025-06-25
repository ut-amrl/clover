import torch
import torch.nn.functional as F
from torch import nn

from models.selavpr.encoder import SelaEncoder, SelaEncoderArgs


class CLOVER_Net(nn.Module):
    def __init__(
        self,
        sela_args: SelaEncoderArgs,
        content_mlp_layers: int = 1,
        content_length: int = 384,
        content_mlp_change_len_last_layer: bool = False,
        use_projection_head_for_learning_content: bool = False,
        projection_head_for_learning_content_layers: int = 1,
        projection_head_for_learning_content_dim: int = 128,
        use_content_head: bool = True,
        **kwargs,
    ):
        super().__init__()
        content_encoder = SelaEncoder(sela_args)

        if not use_content_head:
            assert content_length == content_encoder.getOutputDim(), (
                "Content encoder output length must be equal to content_length if use_content_head is False"
            )

        self.content_encoder = content_encoder

        self.feat_dim = content_length

        self.content_encoder_out_dim = content_encoder.getOutputDim()

        # self.content_head = None
        self.content_head = nn.Identity()

        if use_content_head:
            content_head_layers = []

            if content_mlp_change_len_last_layer:
                for _ in range(content_mlp_layers):
                    content_head_layers.append(
                        nn.Linear(self.content_encoder_out_dim, self.content_encoder_out_dim)
                    )
                    content_head_layers.append(nn.ReLU(inplace=True))
                content_head_layers.append(nn.Linear(self.content_encoder_out_dim, content_length))
            else:
                content_head_layers.append(nn.Linear(self.content_encoder_out_dim, content_length))
                for _ in range(content_mlp_layers):
                    content_head_layers.append(nn.ReLU(inplace=True))
                    content_head_layers.append(nn.Linear(content_length, content_length))

            # Initialize the layers in the heads
            # Use Kaiming normal initialization for the weights and 0 for the biases
            for layer in content_head_layers:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                    nn.init.zeros_(layer.bias)

            self.content_head = nn.Sequential(*content_head_layers)

        self.projection_head_for_learning_content = None

        self.projection_dim = None

        if use_projection_head_for_learning_content:
            self.projection_head_for_learning_content_dim = projection_head_for_learning_content_dim
            layers_projection_head_for_learning_content = []
            for _ in range(projection_head_for_learning_content_layers):
                layers_projection_head_for_learning_content.append(
                    nn.Linear(self.feat_dim, self.feat_dim)
                )
                layers_projection_head_for_learning_content.append(nn.ReLU(inplace=True))
            layers_projection_head_for_learning_content.append(
                nn.Linear(self.feat_dim, self.projection_head_for_learning_content_dim)
            )

            for layer in layers_projection_head_for_learning_content:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                    nn.init.zeros_(layer.bias)

            self.projection_head_for_learning_content = nn.Sequential(
                *layers_projection_head_for_learning_content
            )

        self.default_frozen_params = {}
        if use_projection_head_for_learning_content:
            for name, param in self.projection_head_for_learning_content.named_parameters():
                if not param.requires_grad:
                    self.default_frozen_params["projection_head_for_learning_content"] = name

        for name, param in self.content_head.named_parameters():
            if not param.requires_grad:
                self.default_frozen_params["content_head"] = name

        for name, param in self.content_encoder.named_parameters():
            if not param.requires_grad:
                self.default_frozen_params["content_encoder"] = name

    def freezeWeights(self, freeze_config):
        if freeze_config.freeze_content_encoder:
            print("Freezing content encoder ")
        else:
            print("Unfreezing content encoder (maintaining default frozen weights)")
        for name, param in self.content_encoder.named_parameters():
            if freeze_config.freeze_content_encoder:
                param.requires_grad = False
            else:
                if ("content_encoder" not in self.default_frozen_params) or (
                    name not in self.default_frozen_params["content_encoder"]
                ):
                    param.requires_grad = True

        if freeze_config.freeze_content_head:
            print("Freezing content head")
            for name, param in self.content_head.named_parameters():
                param.requires_grad = False
        else:
            print("Unfreezing content head (maintaining default frozen weights)")
            for name, param in self.content_head.named_parameters():
                if ("content_head" not in self.default_frozen_params) or (
                    name not in self.default_frozen_params["content_head"]
                ):
                    param.requires_grad = True

        if freeze_config.freeze_content_projection_head:
            print("Freezing content projection head")
            for name, param in self.projection_head_for_learning_content.named_parameters():
                param.requires_grad = False
        else:
            print("Unfreezing content projection head (maintaining default frozen weights)")
            for name, param in self.projection_head_for_learning_content.named_parameters():
                if ("projection_head_for_learning_content" not in self.default_frozen_params) or (
                    name not in self.default_frozen_params["projection_head_for_learning_content"]
                ):
                    param.requires_grad = True

    def forward(self, x, mask=None, use_head=True, skip_downstream=False):
        if mask is not None:
            x = torch.cat([x, mask], dim=1)  # Concatenate the image and mask

        content_vec = self.content_head(self.content_encoder(x))

        if (self.projection_head_for_learning_content is not None) and use_head:
            # Normalize the content vector after sending it through the projection head
            content_vec = F.normalize(self.projection_head_for_learning_content(content_vec), dim=1)

        return content_vec

import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """
    Converts an image into a sequence of patch tokens.
    For an image of size (img_size x img_size) and a given patch size,
    it applies a convolution with kernel and stride equal to the patch size.
    """
    def __init__(self, img_size=80, patch_size=4, in_chans=1, embed_dim=16):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size  # number of patches per dimension
        self.num_patches = self.grid_size ** 2
        # A Conv2d layer with kernel and stride equal to patch_size extracts non-overlapping patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x shape: (B, in_chans, img_size, img_size)
        x = self.proj(x)  # shape becomes: (B, embed_dim, grid_size, grid_size)
        x = x.flatten(2)  # flatten the spatial dimensions: (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # rearrange to: (B, num_patches, embed_dim)
        return x

# Example integration into your Transformer model:
class MyTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Create the patch embedding module.
        # Ensure that config has attributes: img_size, patch_size, and hidden_size.
        self.patch_embed = PatchEmbedding(
            img_size=config.img_size,
            patch_size=config.patch_size,  # you need to add this to your config (e.g. 4)
            in_chans=1,                    # change to 3 if using RGB images
            embed_dim=config.hidden_size   # this should match your transformer’s expected dimension
        )

        # Continue with your transformer layers.
        # For example, assuming your transformer blocks are defined elsewhere:
        self.transformer_blocks = nn.ModuleList([
            # ... instantiate your transformer blocks here ...
        ])

        # Optional: a final projection if needed
        self.final_norm = nn.LayerNorm(config.hidden_size)

    def forward(self, x):
        # x is expected to be of shape (B, in_chans, img_size, img_size)
        tokens = self.patch_embed(x)  # now tokens has shape (B, num_patches, hidden_size)

        # If you need to add positional embeddings:
        # pos_embed should be registered as a parameter or computed accordingly.
        # tokens = tokens + self.pos_embed

        # Process tokens with transformer blocks
        for block in self.transformer_blocks:
            tokens = block(tokens)

        tokens = self.final_norm(tokens)
        return tokens

# Example usage:
if __name__ == "__main__":
    # Assuming a configuration object or dictionary with necessary parameters:
    class Config:
        img_size = 80
        patch_size = 4
        hidden_size = 16

    config = Config()
    model = MyTransformer(config)

    # Create a dummy input image (B, 1, 80, 80)
    dummy_input = torch.randn(2, 1, config.img_size, config.img_size)
    output = model(dummy_input)
    print(output.shape)  # expected shape: (B, num_patches, hidden_size) i.e. (2, 400, 16)

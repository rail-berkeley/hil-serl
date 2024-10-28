import flax.linen as nn
from einops import rearrange


class BinaryClassifier(nn.Module):
    pretrained_encoder: nn.Module
    encoder: nn.Module
    network: nn.Module
    enable_stacking: bool = False

    @nn.compact
    def __call__(self, x, train=False, return_encoded=False, classify_encoded=False):
        if return_encoded:
            if self.enable_stacking:
                # Combine stacking and channels into a single dimension
                if len(x.shape) == 4:
                    x = rearrange(x, "T H W C -> H W (T C)")
                if len(x.shape) == 5:
                    x = rearrange(x, "B T H W C -> B H W (T C)")
            x = self.pretrained_encoder(x, train=train)
            return x

        x = self.encoder(x, train=train, is_encoded=classify_encoded)
        x = self.network(x, train=train)
        x = nn.Dense(1)(x).squeeze()
        return x

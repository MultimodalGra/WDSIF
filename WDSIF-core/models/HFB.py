import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange


def to_3d(x):
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type="WithBias"):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == "BiasFree":
            self.body = nn.LayerNorm(dim, elementwise_affine=False)
        else:
            self.body = nn.LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_factor=2.66, bias=False):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_factor)
        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = F.gelu(x)
        x = self.project_out(x)
        return x


class Attention(nn.Module):

    def __init__(self, dim, num_heads, bias=False):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v

        out = rearrange(
            out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w
        )
        out = self.project_out(out)
        return out


class WideTransformerBlock(nn.Module):

    def __init__(
        self, dim, num_heads, ffn_factor=2.66, bias=False, LayerNorm_type="WithBias"
    ):
        super(WideTransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_factor, bias)

        self.wd_conv = nn.Conv2d(dim, dim * 4, kernel_size=1, bias=bias)
        self.wp_conv3 = nn.Conv2d(
            dim, dim, kernel_size=3, padding=1, groups=dim, bias=bias
        )
        self.wp_conv5 = nn.Conv2d(
            dim, dim, kernel_size=5, padding=2, groups=dim, bias=bias
        )
        self.wp_conv7 = nn.Conv2d(
            dim, dim, kernel_size=7, padding=3, groups=dim, bias=bias
        )

        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 8, dim, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x, time_emb=None):
        input_x = x

        wd_out = self.wd_conv(self.norm1(x))
        q, k, v, l = wd_out.chunk(4, dim=1)

        l3 = self.wp_conv3(l)
        l5 = self.wp_conv5(l)
        l7 = self.wp_conv7(l)
        l_combined = l3 + l5 + l7

        attn_out = self.attn(self.norm1(x))

        ca_out = self.channel_attn(l_combined) * l_combined

        x = input_x + attn_out + ca_out

        x = x + self.ffn(self.norm2(x))

        return x


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(
                n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(
                n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(
            in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class HighFrequencyEnhancementBranch(nn.Module):

    def __init__(
        self,
        input_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        heads=[1, 2, 4, 8],
        ffn_factor=2.66,
        bias=False,
        LayerNorm_type="WithBias",
    ):
        super(HighFrequencyEnhancementBranch, self).__init__()

        self.patch_embed = OverlapPatchEmbed(input_channels, dim, bias)

        self.encoder_level1 = nn.Sequential(
            *[
                WideTransformerBlock(
                    dim=dim,
                    num_heads=heads[0],
                    ffn_factor=ffn_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for _ in range(num_blocks[0])
            ]
        )

        self.down1_2 = Downsample(dim)

        self.encoder_level2 = nn.Sequential(
            *[
                WideTransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[1],
                    ffn_factor=ffn_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for _ in range(num_blocks[1])
            ]
        )

        self.down2_3 = Downsample(int(dim * 2**1))

        self.encoder_level3 = nn.Sequential(
            *[
                WideTransformerBlock(
                    dim=int(dim * 2**2),
                    num_heads=heads[2],
                    ffn_factor=ffn_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for _ in range(num_blocks[2])
            ]
        )

        self.down3_4 = Downsample(int(dim * 2**2))

        self.latent = nn.Sequential(
            *[
                WideTransformerBlock(
                    dim=int(dim * 2**3),
                    num_heads=heads[3],
                    ffn_factor=ffn_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for _ in range(num_blocks[3])
            ]
        )

        self.up4_3 = Upsample(int(dim * 2**3))
        self.reduce_chan_level3 = nn.Conv2d(
            int(dim * 2**3), int(dim * 2**2), kernel_size=1, bias=bias
        )

        self.decoder_level3 = nn.Sequential(
            *[
                WideTransformerBlock(
                    dim=int(dim * 2**2),
                    num_heads=heads[2],
                    ffn_factor=ffn_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for _ in range(num_blocks[2])
            ]
        )

        self.up3_2 = Upsample(int(dim * 2**2))
        self.reduce_chan_level2 = nn.Conv2d(
            int(dim * 2**2), int(dim * 2**1), kernel_size=1, bias=bias
        )

        self.decoder_level2 = nn.Sequential(
            *[
                WideTransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[1],
                    ffn_factor=ffn_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for _ in range(num_blocks[1])
            ]
        )

        self.up2_1 = Upsample(int(dim * 2**1))

        self.decoder_level1 = nn.Sequential(
            *[
                WideTransformerBlock(
                    dim=int(dim * 2**1),
                    num_heads=heads[0],
                    ffn_factor=ffn_factor,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for _ in range(num_blocks[0])
            ]
        )

        self.output = nn.Conv2d(
            int(dim * 2**1),
            input_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
        )

    def forward(self, high_freq_inputs):

        inp_enc_level1 = self.patch_embed(high_freq_inputs)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        output = self.output(out_dec_level1)

        return output

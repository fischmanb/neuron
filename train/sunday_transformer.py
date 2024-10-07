import torch
import torch.nn as nn


# TODO sailiency Joseph: have it identify attribution on specific frequencies and replay only those frequencies when doing explainability


def main():
    #d _model is embedding space
    model = nn.Transformer(d_model=2048, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048)
    a = torch.rand((10, 32, 2048))
    tgt = torch.rand((20, 32, 2048))
    out = model(a, tgt)
    print(out.squeeze().shape)


if __name__ == '__main__':
    main()

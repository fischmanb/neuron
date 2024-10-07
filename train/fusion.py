
"""
1. Concatenation (Early Fusion)

You can concatenate all the feature extractions (MFCC, STFT, jitter, shimmer, pitch, etc.) along a new dimension and
feed them into the network together. This is called early fusion, where all features are combined before being passed
into the network.

The network receives all features in a unified form, and the model will automatically learn how to process them together.
"""
import torch
# Example: MFCC, STFT, jitter, shimmer, and pitch tensors
mfcc = torch.randn(1, 13, 100)     # shape: [batch, features, time]
stft = torch.randn(1, 257, 100)    # shape: [batch, features, time]
jitter = torch.randn(1, 1, 100)    # shape: [batch, features, time]
shimmer = torch.randn(1, 1, 100)   # shape: [batch, features, time]
pitch = torch.randn(1, 1, 100)     # shape: [batch, features, time]

# Concatenate along the feature dimension (dim=1)
combined_features = torch.cat((mfcc, stft, jitter, shimmer, pitch), dim=1)  # shape: [1, 273, 100]

# Feed combined_features into the network
output = model(combined_features)


class MultiFeatureNetwork(torch.nn.Module):
    """
    2. Parallel Subnetworks (Mid-Level Fusion)

    Each feature extraction can be processed through its own subnetwork (e.g., separate CNN or RNN layers). The outputs
    of these subnetworks are then concatenated or merged at some point in the network, usually before the final
    classification or regression layers.

    In this approach, each feature is learned through its dedicated subnetwork, and the outputs of these subnetworks
    are merged later on.
    """
    def __init__(self):
        super(MultiFeatureNetwork, self).__init__()
        self.mfcc_net = torch.nn.Conv1d(13, 64, kernel_size=3, padding=1)    # MFCC subnetwork
        self.stft_net = torch.nn.Conv1d(257, 64, kernel_size=3, padding=1)   # STFT subnetwork
        self.jitter_net = torch.nn.Conv1d(1, 16, kernel_size=3, padding=1)   # Jitter subnetwork
        self.shimmer_net = torch.nn.Conv1d(1, 16, kernel_size=3, padding=1)  # Shimmer subnetwork
        self.pitch_net = torch.nn.Conv1d(1, 16, kernel_size=3, padding=1)    # Pitch subnetwork
        self.fc = torch.nn.Linear(64 + 64 + 16 + 16 + 16, 10)  # Fully connected layer (example output size: 10)

    def forward(self, mfcc, stft, jitter, shimmer, pitch):
        mfcc_out = self.mfcc_net(mfcc)
        stft_out = self.stft_net(stft)
        jitter_out = self.jitter_net(jitter)
        shimmer_out = self.shimmer_net(shimmer)
        pitch_out = self.pitch_net(pitch)

        # Flatten and concatenate outputs
        combined_out = torch.cat([mfcc_out.mean(dim=-1), stft_out.mean(dim=-1),
                                  jitter_out.mean(dim=-1), shimmer_out.mean(dim=-1),
                                  pitch_out.mean(dim=-1)], dim=1)
        output = self.fc(combined_out)
        return output


class LateFusionNetwork(torch.nn.Module):
    """
    3. Late Fusion (Ensemble-Like)

    Late fusion treats each feature extraction’s network separately, and the final predictions from each network are
    combined. You could average their outputs, perform a weighted sum, or use a more complex fusion method.

    Here, each subnetwork has a fully connected output, and you combine the predictions from each network at the very
    end.
    """
    def __init__(self):
        super(LateFusionNetwork, self).__init__()
        self.mfcc_net = torch.nn.Sequential(
            torch.nn.Conv1d(13, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1)
        )
        self.stft_net = torch.nn.Sequential(
            torch.nn.Conv1d(257, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1)
        )
        self.jitter_net = torch.nn.Sequential(
            torch.nn.Conv1d(1, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1)
        )
        self.shimmer_net = torch.nn.Sequential(
            torch.nn.Conv1d(1, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1)
        )
        self.pitch_net = torch.nn.Sequential(
            torch.nn.Conv1d(1, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1)
        )

        self.fc = torch.nn.Linear(64 + 64 + 16 + 16 + 16, 10)  # Fully connected layer

    def forward(self, mfcc, stft, jitter, shimmer, pitch):
        mfcc_out = self.mfcc_net(mfcc).view(mfcc.size(0), -1)
        stft_out = self.stft_net(stft).view(stft.size(0), -1)
        jitter_out = self.jitter_net(jitter).view(jitter.size(0), -1)
        shimmer_out = self.shimmer_net(shimmer).view(shimmer.size(0), -1)
        pitch_out = self.pitch_net(pitch).view(pitch.size(0), -1)

        combined_out = torch.cat([mfcc_out, stft_out, jitter_out, shimmer_out, pitch_out], dim=1)
        output = self.fc(combined_out)
        return output


class AttentionFusion(torch.nn.Module):
    """
    4. Attention-Based Fusion

    You can apply an attention mechanism that learns the importance of each feature extraction (MFCC, STFT, etc.)
    dynamically. The model can assign higher weights to more relevant features for the given task.

    The attention mechanism allows the model to learn how much weight to give each feature based on the input data.
    """
    def __init__(self):
        super(AttentionFusion, self).__init__()
        self.mfcc_net = torch.nn.Conv1d(13, 64, kernel_size=3, padding=1)
        self.stft_net = torch.nn.Conv1d(257, 64, kernel_size=3, padding=1)
        self.jitter_net = torch.nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.shimmer_net = torch.nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.pitch_net = torch.nn.Conv1d(1, 16, kernel_size=3, padding=1)

        self.attention = torch.nn.Linear(64 + 64 + 16 + 16 + 16, 1)
        self.fc = torch.nn.Linear(64 + 64 + 16 + 16 + 16, 10)

    def forward(self, mfcc, stft, jitter, shimmer, pitch):
        mfcc_out = self.mfcc_net(mfcc).mean(dim=-1)
        stft_out = self.stft_net(stft).mean(dim=-1)
        jitter_out = self.jitter_net(jitter).mean(dim=-1)
        shimmer_out = self.shimmer_net(shimmer).mean(dim=-1)
        pitch_out = self.pitch_net(pitch).mean(dim=-1)

        combined_out = torch.cat([mfcc_out, stft_out, jitter_out, shimmer_out, pitch_out], dim=1)
        attention_weights = torch.softmax(self.attention(combined_out), dim=1)
        weighted_out = attention_weights * combined_out
        output = self.fc(weighted_out)
        return output


"""
5. Multi-Head Network (Hierarchical Fusion)

This approach treats each feature as part of a hierarchical structure, where the network learns relationships between 
features through multi-head architectures (similar to transformers). Each head focuses on a different feature, and the
outputs are combined later.
"""


"""
6. SIX Whatever this is, combination of the above?
"""
import torch
import torch.nn as nn


class MultiHeadFusionTransformer(nn.Module):
    def __init__(self, d_model=2048, nhead=8, num_encoder_layers=6, dim_feedforward=2048):
        super(MultiHeadFusionTransformer, self).__init__()

        # Transformer Encoders for MFCC and STFT features
        self.mfcc_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward),
            num_layers=num_encoder_layers
        )

        self.stft_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward),
            num_layers=num_encoder_layers
        )

        # Fusion layer: Another transformer encoder to fuse MFCC and STFT outputs
        self.fusion_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward),
            num_layers=num_encoder_layers
        )

        # Fully connected output layer
        self.fc_out = nn.Linear(d_model, d_model)  # Can adjust based on the task

    def forward(self, mfcc, stft):
        # Pass MFCC and STFT through their respective transformer encoders
        mfcc_encoded = self.mfcc_encoder(mfcc)  # Shape: (seq_len_mfcc, batch_size, d_model)
        stft_encoded = self.stft_encoder(stft)  # Shape: (seq_len_stft, batch_size, d_model)

        # Concatenate the outputs of both encoders along the time dimension
        # Shape will now be: (seq_len_mfcc + seq_len_stft, batch_size, d_model)
        combined_features = torch.cat((mfcc_encoded, stft_encoded), dim=0)

        # Pass the combined features through the fusion transformer encoder
        fused_output = self.fusion_transformer(combined_features)

        # Apply fully connected output layer (optional, depends on task)
        out = self.fc_out(fused_output)

        return out


# Example usage
model = MultiHeadFusionTransformer()

# Random MFCC and STFT tensors
# MFCC input: (sequence_length, batch_size, feature_dim)
mfcc = torch.rand((10, 32, 2048))  # Sequence length 10 for MFCC
stft = torch.rand((20, 32, 2048))  # Sequence length 20 for STFT

# Forward pass
output = model(mfcc, stft)
print(output.squeeze().shape)
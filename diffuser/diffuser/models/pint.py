"""
AdaLNPINTDenoiser
-----------------

This module implements a denoising network based on an Adaptive Layer-Normalized
Particle Interaction Transformer (AdaLNParticleTransformer). It is designed for processing
particle-based state inputs with corresponding actions and a time-conditioning signal.
The network projects raw features into a latent space, applies a transformer to model
interactions among particles (and actions), and finally decodes the representations back
to the original feature dimensions.

Usage Example:
    model = AdaLNPINTDenoiser(
        features_dim=10, action_dim=3, hidden_dim=256, projection_dim=256,
        n_head=8, n_layer=6, block_size=50, dropout=0.1,
        predict_delta=False, positional_bias=True, max_particles=4,
        learned_sinusoidal_cond=False, random_fourier_features=False,
        learned_sinusoidal_dim=16, multiview=False
    )
    # x: [batch_size, time_steps, action_dim + particle_feature_dim]
    # t: [batch_size] (e.g., time indices)
    out = model(x, cond=None, time=t)
"""

import torch
from torch import nn
from diffuser.models.transformer_modules import (
    AdaLNParticleTransformer,
    SinusoidalPosEmb,
    RandomOrLearnedSinusoidalPosEmb,
)

class AdaLNPINTDenoiser(nn.Module):
    """
    AdaLNPINTDenoiser

    Implements a denoising model based on an Adaptive Layer Normalized Particle Interaction
    Transformer. It processes sequences of particle state features concatenated with action
    information and conditioned on a time signal.

    Parameters:
        features_dim (int): Dimensionality of each particle's feature vector.
        action_dim (int): Dimensionality of the action vector.
        hidden_dim (int): Hidden dimension used in projection layers.
        projection_dim (int): Dimension of the latent space in the transformer.
        n_head (int): Number of attention heads in the transformer.
        n_layer (int): Number of transformer layers.
        block_size (int): Time horizon (number of time steps).
        dropout (float): Dropout probability for transformer components.
        predict_delta (bool): If True, the model predicts a delta change rather than an absolute value.
        positional_bias (bool): If True, applies positional bias in the transformer.
        max_particles (int or None): Maximum number of particles (for relative positional bias).
        learned_sinusoidal_cond (bool): If True, use a learned sinusoidal embedding for time conditioning.
        random_fourier_features (bool): If True, use fixed random Fourier features.
        learned_sinusoidal_dim (int): Dimensionality for the learned sinusoidal (or Fourier) features.
        multiview (bool): If True, use separate encodings for multi-view particle inputs.
    """
    def __init__(self, features_dim=2, action_dim=3, hidden_dim=256, projection_dim=256,
                 n_head=8, n_layer=6, block_size=50, dropout=0.1,
                 predict_delta=False, positional_bias=True, max_particles=4,
                 learned_sinusoidal_cond=False, random_fourier_features=False,
                 learned_sinusoidal_dim=16, multiview=False, **kwargs):
        super(AdaLNPINTDenoiser, self).__init__()

        self.features_dim = features_dim
        self.action_dim = action_dim
        self.predict_delta = predict_delta
        self.projection_dim = projection_dim
        self.max_particles = max_particles
        self.multiview = multiview
        # block_size is the time horizon

        # Define an intermediate time embedding dimension.
        time_dim = projection_dim * 4

        # Decide whether to use random/learned Fourier features for time conditioning.
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features
        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            # Fourier feature output is concatenated with the original scalar, so add 1.
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(projection_dim)
            fourier_dim = projection_dim

        # Time embedding network.
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, projection_dim)
        )

        # Particle feature projection network.
        self.particle_projection = nn.Sequential(
            nn.Linear(self.features_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.projection_dim)
        )
        # Action feature projection network.
        self.action_projection = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.projection_dim)
        )

        # Instantiate the AdaLN Particle Transformer.
        self.particle_transformer = AdaLNParticleTransformer(
            self.projection_dim, n_head, n_layer, block_size, self.projection_dim,
            attn_pdrop=dropout, resid_pdrop=dropout,
            hidden_dim_multiplier=4,
            positional_bias=positional_bias,
            activation='gelu', max_particles=max_particles
        )

        # Decoder networks for particle and action outputs.
        self.particle_decoder = nn.Sequential(
            nn.Linear(self.projection_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.features_dim)
        )
        self.action_decoder = nn.Sequential(
            nn.Linear(self.projection_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.action_dim)
        )
        # Particle encoding: either shared or view-specific for multi-view inputs.
        if self.multiview:
            self.view1_encoding = nn.Parameter(0.02 * torch.randn(1, 1, 1, projection_dim))
            self.view2_encoding = nn.Parameter(0.02 * torch.randn(1, 1, 1, projection_dim))
        else:
            self.particle_encoding = nn.Parameter(0.02 * torch.randn(1, 1, 1, projection_dim))
        
        # Learnable action encoding.
        self.action_encoding = nn.Parameter(0.02 * torch.randn(1, 1, projection_dim))

    def forward(self, x, cond, time, return_attention=False):
        """
        Forward pass for the denoiser.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, T, action_dim + particle_feature_dim].
                The first action_dim elements are the action features, and the rest are particle features.
            cond: (Unused) Additional conditioning (reserved for future use).
            time (torch.Tensor): Tensor of time indices with shape [batch_size]. These are embedded via time_mlp.
            return_attention (bool): If True, returns the attention weights along with the output.

        Returns:
            torch.Tensor or tuple: 
                - If return_attention is False: output tensor of shape [batch_size, T, output_dim],
                  where output_dim = action_dim + (n_particles * features_dim).
                - If return_attention is True: (output, attention_dict)
        """
        # ---------------------------------------------------------------------
        # Reshape input: separate actions and particle features.
        # x: [bs, T, action_dim + particle_feature_dim]
        bs, T, f = x.size()
        actions = x[:, :, :self.action_dim]  # [bs, T, action_dim]
        # Reshape remaining features into particles of shape [bs, T, n_particles, features_dim].
        x_particles = x[:, :, self.action_dim:].view(bs, T, -1, self.features_dim)

        # ---------------------------------------------------------------------
        # Project actions and particles.
        action_particle = self.action_projection(actions)  # [bs, T, projection_dim]
        action_particle = action_particle + self.action_encoding.repeat(bs, T, 1)
        state_particles = self.particle_projection(x_particles)  # [bs, T, n_particles, projection_dim]

        if self.multiview:
            n_particles = state_particles.size(2) // 2
            particles_view1 = state_particles[:, :, :n_particles, :] + self.view1_encoding.repeat(bs, T, n_particles, 1)
            particles_view2 = state_particles[:, :, n_particles:, :] + self.view2_encoding.repeat(bs, T, n_particles, 1)
            new_state_particles = torch.cat([particles_view1, particles_view2], dim=2)
        else:
            new_state_particles = state_particles + self.particle_encoding.repeat(bs, T, state_particles.size(2), 1)

        # ---------------------------------------------------------------------
        # Prepare transformer input.
        # Concatenate the action token with the particle tokens.
        # Resulting shape: [bs, T, n_tokens, projection_dim], where n_tokens = 1 + n_particles.
        x_cat = torch.cat([action_particle.unsqueeze(2), new_state_particles], dim=2)

        # Time embedding: project time indices and add to all tokens.
        t_embed = self.time_mlp(time)  # [bs, projection_dim]
        x_proj = x_cat + t_embed[:, None, None, :]  # Broadcast addition.

        # Permute to match transformer input shape: [bs, n_tokens, T, projection_dim]
        x_proj = x_proj.permute(0, 2, 1, 3)

        # ---------------------------------------------------------------------
        # Apply the particle transformer.
        if return_attention:
            particles_trans, attention_dict = self.particle_transformer(x_proj, action_particle, t_embed,
                                                                         return_attention=return_attention)
        else:
            particles_trans = self.particle_transformer(x_proj, action_particle, t_embed)
        # Permute back to [bs, T, n_tokens, projection_dim].
        particles_trans = particles_trans.permute(0, 2, 1, 3)

        # ---------------------------------------------------------------------
        # Decode transformer output.
        action_decoder_out = self.action_decoder(particles_trans[:, :, 0, :])  # [bs, T, action_dim]
        particle_decoder_out = self.particle_decoder(particles_trans[:, :, 1:, :])
        particle_decoder_out = particle_decoder_out.view(bs, T, -1)  # Flatten particle outputs.
        # Concatenate action and particle outputs.
        x_out = torch.cat([action_decoder_out, particle_decoder_out], dim=-1)

        if return_attention:
            return x_out, attention_dict
        else:
            return x_out

# ------------------------------------------------------------------------------
# Test block
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    batch_size = 32
    timessteps = 5
    model = AdaLNPINTDenoiser(features_dim=10, action_dim=3, hidden_dim=256, projection_dim=256,
                        n_head=8, n_layer=6, block_size=timessteps, dropout=0.1,
                        predict_delta=False, positional_bias=False, max_particles=None,
                        learned_sinusoidal_cond=False, random_fourier_features=False, learned_sinusoidal_dim=16)
    in_particles = torch.randn(batch_size, timessteps, 240)
    actions = torch.randn(batch_size, timessteps, 3)
    t = torch.randint(0, 1000, (batch_size,), device=in_particles.device).long()

    # Concatenate actions and particle features.
    x = torch.cat([actions, in_particles], dim=-1)
    model_out = model(x, cond=None, time=t, return_attention=False)
    print("Output shape:", model_out.shape)

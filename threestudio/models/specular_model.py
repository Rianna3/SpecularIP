import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class ASGRender(nn.Module):
    """
    Approximate Specular response using a set of anisotropic spherical Gaussians (ASGs).
    asg_params: (N, L, 4): [amplitude a, sharpness lambda, theta, phi]
    viewdirs: (N, 3) unit vectors pointing from point to camera (-ray direction)
    normals: (N, 3) unit surface normals (approx/proxy)

    Returns per-lobe responses: (N, L)
    """
    def __init__(self):
        super().__init__()

    def forward(self, viewdirs: torch.Tensor, asg_params: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
        device = viewdirs.device
        N, L, _ = asg_params.shape
        # Unpack parameters
        a = F.softplus(asg_params[..., 0])            # (N, L)
        lam = F.softplus(asg_params[..., 1]) + 1e-4   # (N, L) positive sharpness
        theta = torch.tanh(asg_params[..., 2]) * (0.5 * torch.pi)  # (-pi/2, pi/2)
        phi = torch.tanh(asg_params[..., 3]) * torch.pi           # (-pi, pi)

        # Build lobe mean direction mu in local frame around the reflection dir
        # Compute reflection direction R = 2*(V·N)*N - V, where V = viewdirs
        V = viewdirs  # (N,3) already unit
        Nn = F.normalize(normal, dim=-1)
        R = F.normalize(2.0 * (V * Nn).sum(dim=-1, keepdim=True) * Nn - V, dim=-1)  # (N,3)

        # Construct an orthonormal basis around R: (T,B,R)
        # Pick arbitrary vector to build tangent
        up = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=R.dtype).expand_as(R)
        t = F.normalize(torch.cross(up, R), dim=-1)
        mask = (t.norm(dim=-1, keepdim=True) < 1e-4)
        t = torch.where(mask, F.normalize(torch.cross(torch.tensor([1.0,0.0,0.0], device=device, dtype=R.dtype).expand_as(R), R), dim=-1), t)
        b = F.normalize(torch.cross(R, t), dim=-1)

        # Spherical to Cartesian in this local frame
        sin_theta = torch.sin(theta)
        mu_local = torch.stack([sin_theta * torch.cos(phi), sin_theta * torch.sin(phi), torch.cos(theta)], dim=-1)  # (N,L,3)
        # Map to world: mu = mu_x * t + mu_y * b + mu_z * R
        tL = t[:, None, :]  # (N,1,3)
        bL = b[:, None, :]
        RL = R[:, None, :]
        mu = mu_local[..., 0:1] * tL + mu_local[..., 1:2] * bL + mu_local[..., 2:3] * RL  # (N,L,3)
        mu = F.normalize(mu, dim=-1)

        # Evaluate ASG response w = a * exp(lam * (dot(R, mu) - 1))
        # Higher when mu aligns with reflection direction
        dot = (RL * mu).sum(dim=-1)  # (N,L)
        w = a * torch.exp(lam * (dot - 1.0))
        return w  # (N,L)

class SpecularNetwork(nn.Module):
    def __init__(self, asg_dim: int, num_theta: int = 3, num_phi: int = 2, hidden_dim: int = 64):
        super().__init__()
        self.num_lobes = num_theta * num_phi
        self.proj = nn.Linear(asg_dim, self.num_lobes * 4)
        self.asg = ASGRender()
        self.mlp = nn.Sequential(
            nn.Linear(self.num_lobes, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid(),  # RGB in [0,1]
        )
        # Initialize last layer bias low to start with dark specular
        if isinstance(self.mlp[2], nn.Linear):
            if isinstance(self.mlp[2].bias, torch.Tensor):
                nn.init.constant_(self.mlp[2].bias, -2.0)

    def forward(self, asg_feat: torch.Tensor, viewdir: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
        # asg_feat: (N, D), viewdir: (N,3), normal: (N,3)
        params = self.proj(asg_feat)  # (N, L*4)
        params = params.view(asg_feat.shape[0], self.num_lobes, 4)
        responses = self.asg(viewdir, params, normal)  # (N,L)
        rgb = self.mlp(responses)  # (N,3)
        return rgb 
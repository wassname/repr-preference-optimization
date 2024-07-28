import torch
from torch import nn, Tensor
from typing import Tuple
from jaxtyping import Float

class SVDDecomposer:
    """
    Decompose hidden states into the internal and external components using SVD.

    These are the component that are not projected onto the LLM output and are
    """
    def __init__(self, W: Float[Tensor, 'hs vocab_size'], epsilon: float = 1e-12):
        dtype = W.dtype
        W = W.float()
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        self.S = S.to(dtype)
        self.Vt = Vt.to(dtype)
        
        self.S = S
        self.Vt = Vt
        self.epsilon = epsilon

    def decompose(self, hs: Float[Tensor, "batch layers tokens hidden_size"]) -> Tuple[Float[Tensor, "batch layers tokens hidden_size"], Float[Tensor, "batch layers tokens hidden_size"]]:
        original_shape = hs.shape

        # flatten
        hs_flat = hs.view(-1, original_shape[-1])

        # dtype and device
        hs_flat = hs_flat.to(self.Vt.dtype).to(self.Vt.device)
        
        # Project onto the right singular vectors
        projection = self.Vt @ hs_flat.T
        
        # Apply a soft thresholding
        S_inv = self.S / (self.S**2 + self.epsilon)
        soft_threshold = torch.sign(projection) * torch.max(torch.abs(projection) - self.epsilon, torch.zeros_like(projection))
        
        # Reconstruct
        hs_ext_flat = (self.Vt.T @ (S_inv.unsqueeze(1) * soft_threshold)).T
        hs_int_flat = hs_flat - hs_ext_flat
        
        # Return to original shape
        hs_external = hs_ext_flat.view(original_shape)
        hs_internal = hs_int_flat.view(original_shape)
        
        return hs_internal, hs_external

    def estimate_error(self, hs: Float[Tensor, "batch layers tokens hidden_size"]) -> float:
        hs_r, _ = self.decompose(hs)
        relative_error = torch.norm(hs_r) / torch.norm(hs)
        return relative_error.item()

    def get_condition_number(self) -> float:
        return (self.S[0] / self.S[-1]).item()
    
    # def reconstruct_from_residual(self, hs_r: Float[Tensor, "batch layers tokens hidden_size"]) -> Float[Tensor, "batch layers tokens hidden_size"]:
    #     hs_r 
    #     return hs_h

# # Usage
# decomposer = OptimizedSVDDecomposer(lm_head, epsilon=1e-12)

# # For R matrix
# hs_r_R, hs_h_R = decomposer.decompose(R)
# error_R = decomposer.estimate_error(R)

# # For C matrix
# hs_r_C, hs_h_C = decomposer.decompose(C)
# error_C = decomposer.estimate_error(C)

# print(f"R matrix - Estimated relative error: {error_R}")
# print(f"C matrix - Estimated relative error: {error_C}")
# print(f"Frobenius norm of difference in residuals: {torch.norm(hs_r_R - hs_r_C)}")
# print(f"Relative difference in residuals: {torch.norm(hs_r_R - hs_r_C) / torch.norm(hs_r_R)}")

# # Additional analysis
# print(f"Frobenius norm of difference between R and C: {torch.norm(R - C)}")
# print(f"Relative difference between R and C: {torch.norm(R - C) / torch.norm(R)}")
# print(f"Condition number: {decomposer.get_condition_number()}")

# # Analyze the differences in the head space
# hs_h_diff = torch.norm(hs_h_R - hs_h_C) / torch.norm(hs_h_R)
# print(f"Relative difference in head space projections: {hs_h_diff}")

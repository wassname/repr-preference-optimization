import torch
from torch import nn, Tensor
from typing import Tuple
from jaxtyping import Float

def soft_svd(W, quantile=0.1, full_matrices=False):
    """Instead of the hard cutoff in traditional SVD, soft SVD uses a continuous weighting of singular values. This can be useful for smoothly interpolating between low-rank approximations and the full matrix."""
    U, S, Vt = torch.linalg.svd(W, full_matrices=full_matrices)

    tau = torch.quantile(S, quantile)
    # print(m, S, tau)
    m = (S.abs() > tau).float().mean()
    print(f"Soft SVD: {m:.2%} of singular values kept, with tau={tau:.2f}, Smean={S.abs().mean():.2f}, Smax={S.max():.2f}, Smin={S.min():.2f}")
    
    # Soft thresholding function
    S_soft = torch.sign(S) * (torch.maximum(torch.abs(S) - tau, torch.zeros_like(S)) + tau)
    
    # Normalize to keep total "energy" constant
    S_soft = S_soft * (torch.norm(S) / torch.norm(S_soft))
    
    return U, S_soft, Vt

def mse(x, y=None, dim=None):
    if y is None:
        y = torch.zeros_like(x)
    return ((x - y) ** 2).mean(dim)


class SoftSVDDecomposer:
    """
    Decompose hidden states into the internal and external components using SVD.

    These are the component that are not projected onto the LLM output and are
    """
    def __init__(self, W: Float[Tensor, 'hs vocab_size'], full_matrices=False, quantile=0.1):
        dtype = W.dtype
        dtype = torch.float32
        self.W = W.float()
        # in LORA they don't use full matrices https://github.dev/huggingface/peft/blob/46f78978f1087d9c0351ade0ec28c1b4acd3ae2c/src/peft/tuners/lora/layer.py#L202
        U, S, Vt = soft_svd(self.W, quantile=quantile, full_matrices=full_matrices)

        a,b = Vt.shape
        assert a==b, 'Vt should be square,try transposing W'

        self.U = U.to(dtype).detach()
        self.S = S.to(dtype).detach()
        self.Vt = Vt.to(dtype).detach()


    def __call__(self, hs: Float[Tensor, "batch layers tokens hidden_size"]) -> Tuple[Float[Tensor, "batch layers tokens hidden_size"], Float[Tensor, "batch layers tokens hidden_size"]]:
        original_shape = hs.shape

        def preshape(hs):
            return hs.view(-1, original_shape[-1]).to(self.Vt.device)
                # .to(self.Vt.dtype)

        hs_flat = preshape(hs)
        
        # Project onto the right singular vectors
        assert self.Vt.shape[0]==hs_flat.shape[-1]
        assert self.Vt.shape[1]==hs_flat.shape[-1]
        # hs_projected = hs_flat @ self.Vt.T @ self.Vt  # projection onto W's row space
        hs_projected = hs @ self.Vt.T @ torch.diag(self.S) @ self.Vt

        # hs_ov = hs @ U @ U.T  # projection onto W's row space
        assert torch.isfinite(hs_projected).all()
        
        return hs_projected.view(original_shape)

    def test(self, hs):
        hs_ov = self(hs)
        hs_r = hs - hs_ov

        original_projection = mse(self.W @ hs.T)
        output_projection = mse(self.W @ hs_ov.T)
        internal_projection = mse(self.W @ hs_r.T)

        print(f"Original: {original_projection}")
        print(f"Output: {output_projection}")
        print(f"Internal: {internal_projection}")

        # the internal projection should be small
        assert torch.allclose(internal_projection, internal_projection*0, atol=10)

        # the output projection should be the same as the original
        assert torch.allclose(original_projection, output_projection, atol=19)



class SVDDecomposer:
    """
    Decompose hidden states into the internal and external components using SVD.

    These are the component that are not projected onto the LLM output and are
    """
    def __init__(self, W: Float[Tensor, 'hs vocab_size'], full_matrices=False):
        dtype = W.dtype
        dtype = torch.float32
        self.W = W.float()
        # in LORA they don't use full matrices https://github.dev/huggingface/peft/blob/46f78978f1087d9c0351ade0ec28c1b4acd3ae2c/src/peft/tuners/lora/layer.py#L202
        U, S, Vt = torch.linalg.svd(self.W, full_matrices=full_matrices)

        a,b = Vt.shape
        assert a==b, 'Vt should be square,try transposing W'

        self.U = U.to(dtype).detach()
        self.S = S.to(dtype).detach()
        self.Vt = Vt.to(dtype).detach()


    def __call__(self, hs: Float[Tensor, "batch layers tokens hidden_size"]) -> Tuple[Float[Tensor, "batch layers tokens hidden_size"], Float[Tensor, "batch layers tokens hidden_size"]]:
        original_shape = hs.shape

        def preshape(hs):
            return hs.view(-1, original_shape[-1]).to(self.Vt.dtype).to(self.Vt.device)

        hs_flat = preshape(hs)
        
        # Project onto the right singular vectors
        assert self.Vt.shape[0]==hs_flat.shape[-1]
        assert self.Vt.shape[1]==hs_flat.shape[-1]
        hs_ov = hs_flat @ self.Vt.T @ self.Vt  # projection onto W's row space

        # hs_ov = hs @ U @ U.T  # projection onto W's row space
        assert torch.isfinite(hs_ov).all()
        
        return hs_ov.view(original_shape)

    def test(self, hs):
        hs_ov = self(hs)
        hs_r = hs - hs_ov

        original_projection = mse(self.W @ hs.T)
        output_projection = mse(self.W @ hs_ov.T)
        internal_projection = mse(self.W @ hs_r.T)

        print(f"Original: {original_projection}")
        print(f"Output: {output_projection}")
        print(f"Internal: {internal_projection}")

        # the internal projection should be small
        assert torch.allclose(internal_projection, internal_projection*0, atol=10)

        # the output projection should be the same as the original
        assert torch.allclose(original_projection, output_projection, atol=19)



class DualSVDDecomposer:
    """
    d = DualSVDDecomposer(model.get_embedding_weights(), model.lm_head.weight)
    d(hs)
    """
    def __init__(self, W_in: Float[Tensor, 'vocab_size hidden_size'], W_out: Float[Tensor, 'hidden_size vocab_size'], full_matrices=False, quantile=0.1):
        self.decomposer_in = SoftSVDDecomposer(W_in, full_matrices=full_matrices, quantile=quantile)
        self.decomposer_out = SoftSVDDecomposer(W_out, full_matrices=full_matrices, quantile=quantile)

    def __call__(self, hs: Float[Tensor, "batch layers tokens hidden_size"]) -> Tuple[Float[Tensor, "batch layers tokens hidden_size"], Float[Tensor, "batch layers tokens hidden_size"], Float[Tensor, "batch layers tokens hidden_size"]]:
        hs_external_in = self.decomposer_in(hs)
        hs_external_out = self.decomposer_out(hs)
        
        hs_io = hs_external_in + hs_external_out - (hs_external_in * hs_external_out).sum(dim=-1, keepdim=True) * hs_external_out / (hs_external_out * hs_external_out).sum(dim=-1, keepdim=True)
        hs_io = hs_io.to(hs.dtype).to(hs.device)
        
        return hs_io
    
    def test(self, hs):
        print('decomposer_in')
        self.decomposer_in.test(hs)
        print('decomposer_out')
        self.decomposer_out.test(hs)

        # now also test combined
        print('combined')
        hs_io = self(hs)
        hs_r = hs - hs_io
        WE = self.decomposer_in.W 
        WO = self.decomposer_out.W
        original_projection = mse((WE @ hs.T).T @ WO)
        output_projection = mse((WE @ hs_io.T).T @ WO)
        internal_projection = mse((WE @ hs_r.T).T @ WO)

        print(f"Original: {original_projection}")
        print(f"Output: {output_projection}")
        print(f"Internal: {internal_projection}")

        torch.testing.assert_close(original_projection, output_projection, rtol=1e-3, atol=19,
                                    msg="Output projection should be close too the original")

        # the internal projection should be small
        torch.testing.assert_close(internal_projection, torch.tensor(0.), rtol=1e-3, atol=10,
                                    msg="Internal projection should be close to zero")

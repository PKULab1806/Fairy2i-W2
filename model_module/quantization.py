import torch
import torch.nn as nn
import math

@torch.no_grad()
def quantize_complex_tensor(w_real: torch.Tensor, w_imag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply PhaseQuant logic to complex weight tensors"""
    phase = torch.angle(w_real + 1j * w_imag)

    real_pos = (phase >= -math.pi / 4) & (phase < math.pi / 4)
    real_neg = (phase >= 3 * math.pi / 4) | (phase < -3 * math.pi / 4)
    imag_pos = (phase >= math.pi / 4) & (phase < 3 * math.pi / 4)
    imag_neg = (phase >= -3 * math.pi / 4) & (phase < -math.pi / 4)

    mask_real = real_pos | real_neg
    mask_imag = imag_pos | imag_neg

    s_re = w_real[mask_real].abs().mean() if mask_real.any() else torch.tensor(0.0, device=w_real.device)
    s_im = w_imag[mask_imag].abs().mean() if mask_imag.any() else torch.tensor(0.0, device=w_imag.device)
    
    s_re = torch.clamp(s_re, min=1e-6)
    s_im = torch.clamp(s_im, min=1e-6)
    if torch.isnan(s_re) or torch.isinf(s_re): s_re = torch.tensor(1e-6, device=w_real.device)
    if torch.isnan(s_im) or torch.isinf(s_im): s_im = torch.tensor(1e-6, device=w_imag.device)

    qw_real = torch.zeros_like(w_real)
    qw_imag = torch.zeros_like(w_imag)
    
    qw_real[real_pos] = 1.0
    qw_real[real_neg] = -1.0
    qw_imag[imag_pos] = 1.0
    qw_imag[imag_neg] = -1.0

    qw_real_scaled = qw_real * s_re
    qw_imag_scaled = qw_imag * s_im
    return qw_real_scaled.to(w_real.dtype), qw_imag_scaled.to(w_imag.dtype)

def apply_complex_inspired_quantization(model: nn.Module):
    """Apply complex-inspired quantization to real-valued model"""
    print("Applying complex-inspired quantization (PhaseQuant-based)...")
    
    @torch.no_grad()
    def quantize_linear_layer(module: nn.Linear):
        A = module.weight.data
        if A.shape[0] % 2 != 0 or A.shape[1] % 2 != 0:
            print(f"  -> Skipping layer (non-even dimensions): {A.shape}")
            return

        n, m = A.shape[0] // 2, A.shape[1] // 2
        A11, A12 = A[:n, :m], A[:n, m:]
        A21, A22 = A[n:, :m], A[n:, m:]

        U_re = 0.5 * (A11 + A22)
        U_im = 0.5 * (A21 - A12)
        W_re = 0.5 * (A11 - A22)
        W_im = 0.5 * (A12 + A21)

        U_re_q, U_im_q = quantize_complex_tensor(U_re, U_im)
        W_re_q, W_im_q = quantize_complex_tensor(W_re, W_im)

        A11_q = W_re_q + U_re_q
        A12_q = W_im_q - U_im_q
        A21_q = W_im_q + U_im_q
        A22_q = -W_re_q + U_re_q

        A_quant_top = torch.cat([A11_q, A12_q], dim=1)
        A_quant_bottom = torch.cat([A21_q, A22_q], dim=1)
        A_quant = torch.cat([A_quant_top, A_quant_bottom], dim=0)

        module.weight.data = A_quant.to(A.dtype)

    model.apply(lambda module: quantize_linear_layer(module) if isinstance(module, nn.Linear) else None)
    print("Complex-inspired quantization completed.")
    return model

def apply_bitnet_quantization(model: nn.Module):
    """Apply BitNet 1-bit quantization to real-valued model"""
    print("Applying BitNet (true 1-bit, affine) quantization to real-valued model...")
    
    @torch.no_grad()
    def quantize_linear_layer(module: nn.Linear):
        scale = module.weight.data.abs().mean()
        alpha = module.weight.data.mean()
        centered_weights = module.weight.data - alpha
        binarized_weights = torch.where(centered_weights > 0, 1.0, -1.0)
        module.weight.data = binarized_weights.to(module.weight.data.dtype) * scale

    model.apply(lambda module: quantize_linear_layer(module) if isinstance(module, nn.Linear) else None)
    print("BitNet quantization completed.")
    return model

def apply_bitnet_1_58bit_quantization_standard(model: nn.Module):
    """Apply BitNet 1.58-bit quantization to real-valued model (quantize to {-1, 0, +1})"""
    print("Applying BitNet 1.58-bit (absmean threshold) quantization to real-valued model...")
    
    @torch.no_grad()
    def quantize_linear_layer(module: nn.Linear):
        W = module.weight.data
        gamma = W.abs().mean()
        W_normalized = W / (gamma + 1e-5)
        W_quantized = torch.clamp(torch.round(W_normalized), -1.0, 1.0)
        module.weight.data = W_quantized.to(W.dtype) * gamma
    
    model.apply(lambda module: quantize_linear_layer(module) if isinstance(module, nn.Linear) else None)
    print("BitNet 1.58-bit (absmean threshold) quantization completed.")
    return model

def apply_bitnet_1_58bit_quantization_variant(model: nn.Module, threshold: float = 0.5):
    """Apply BitNet 1.58-bit quantization to real-valued model (quantize to {-1, 0, +1})"""
    print("Applying BitNet 1.58-bit (ternary) quantization to real-valued model...")
    
    @torch.no_grad()
    def quantize_linear_layer(module: nn.Linear):
        gamma = module.weight.data.abs().mean()
        normalized_weights = module.weight.data / (gamma + 1e-5)
        adaptive_threshold = threshold
        ternary_weights = torch.zeros_like(normalized_weights)
        ternary_weights[normalized_weights > adaptive_threshold] = 1.0
        ternary_weights[normalized_weights < -adaptive_threshold] = -1.0
        module.weight.data = ternary_weights.to(module.weight.data.dtype) * gamma
    
    model.apply(lambda module: quantize_linear_layer(module) if isinstance(module, nn.Linear) else None)
    print("BitNet 1.58-bit quantization completed.")
    return model

def minmax_1bit_quantize_dequantize(w: torch.Tensor) -> torch.Tensor:
    """Apply 1-bit Min-Max quantization and dequantization to weight tensor"""
    min_val = w.min()
    max_val = w.max()
    scale = (max_val - min_val) / 1.0
    zero_point = min_val

    if abs(scale) < 1e-9:
        return w

    quantized_w = torch.round((w - zero_point) / scale)
    dequantized_w = quantized_w * scale + zero_point
    
    return dequantized_w.to(w.dtype)

def apply_minmax_1bit_quantization(model: nn.Module):
    """Apply Min-Max 1-bit quantization to real-valued model"""
    print("Applying Min-Max (1-bit) quantization to real-valued model...")
    
    @torch.no_grad()
    def quantize_linear_layer(module: nn.Linear):
        module.weight.data = minmax_1bit_quantize_dequantize(module.weight.data)

    model.apply(lambda module: quantize_linear_layer(module) if isinstance(module, nn.Linear) else None)
    
    print("Min-Max 1-bit quantization completed.")
    return model

def symmetric_minmax_1bit_quantize_dequantize(w: torch.Tensor) -> torch.Tensor:
    """Apply symmetric 1-bit Min-Max quantization to weight tensor (quantize to {-1, 1})"""
    max_abs = w.abs().max()
    scale = max_abs

    if scale < 1e-9:
        return w

    quantized_w = (w / scale).sign()
    dequantized_w = quantized_w * scale
    
    return dequantized_w.to(w.dtype)

def apply_symmetric_minmax_1bit_quantization(model: nn.Module):
    """Apply symmetric Min-Max 1-bit quantization to real-valued model"""
    print("Applying symmetric Min-Max (1-bit, to {-1, 1}) quantization to real-valued model...")
    
    @torch.no_grad()
    def quantize_linear_layer(module: nn.Linear):
        module.weight.data = symmetric_minmax_1bit_quantize_dequantize(module.weight.data)

    model.apply(lambda module: quantize_linear_layer(module) if isinstance(module, nn.Linear) else None)
    
    print("Symmetric Min-Max 1-bit quantization completed.")
    return model

class BitNetQuantSTE(torch.autograd.Function):
    """BitNet STE: quantize in forward, pass gradients in backward"""
    @staticmethod
    def forward(ctx, w):
        scale = w.abs().mean()
        alpha = w.mean()
        centered_w = w - alpha
        binarized_w = torch.where(centered_w > 0, 1.0, -1.0).to(w.dtype)
        quantized_w = binarized_w * scale
        return quantized_w

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class BitNet1_58QuantSTE(torch.autograd.Function):
    """BitNet 1.58-bit STE: quantize to {-1, 0, +1}, pass gradients in backward"""
    @staticmethod
    def forward(ctx, w):
        gamma = w.abs().mean()
        w_normalized = w / (gamma + 1e-5)
        w_quantized = torch.clamp(torch.round(w_normalized), -1.0, 1.0)
        quantized_w = (w_quantized * gamma).to(w.dtype)
        return quantized_w
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class PhaseQuantSTE(torch.autograd.Function):
    """Complex-Phase STE: quantize in forward, pass gradients in backward"""
    @staticmethod
    def forward(ctx, w_real, w_imag):
        phase = torch.angle(w_real + 1j * w_imag)
        
        real_pos = (phase >= -math.pi / 4) & (phase < math.pi / 4)
        real_neg = (phase >= 3 * math.pi / 4) | (phase < -3 * math.pi / 4)
        imag_pos = (phase >= math.pi / 4) & (phase < 3 * math.pi / 4)
        imag_neg = (phase >= -3 * math.pi / 4) & (phase < -math.pi / 4)

        mask_real = real_pos | real_neg
        mask_imag = imag_pos | imag_neg
        
        s_re = w_real[mask_real].abs().mean() if mask_real.any() else torch.tensor(0.0, device=w_real.device)
        s_im = w_imag[mask_imag].abs().mean() if mask_imag.any() else torch.tensor(0.0, device=w_imag.device)
        
        s_re = torch.clamp(s_re, min=1e-6)
        s_im = torch.clamp(s_im, min=1e-6)
        
        qw_real = torch.zeros_like(w_real)
        qw_imag = torch.zeros_like(w_imag)
        
        qw_real[real_pos] = 1.0
        qw_real[real_neg] = -1.0
        qw_imag[imag_pos] = 1.0
        qw_imag[imag_neg] = -1.0
        
        qw_real_scaled = qw_real * s_re
        qw_imag_scaled = qw_imag * s_im
        
        return qw_real_scaled.to(w_real.dtype), qw_imag_scaled.to(w_imag.dtype)

    @staticmethod
    def backward(ctx, grad_w_real, grad_w_imag):
        return grad_w_real, grad_w_imag

class PhaseQuantSTE_V2(torch.autograd.Function):
    """Two-step residual quantization"""

    @staticmethod
    def forward(ctx, w_real: torch.Tensor, w_imag: torch.Tensor):
        qw_real_o1, qw_imag_o1 = PhaseQuantSTE.apply(w_real, w_imag)
        error_real = w_real - qw_real_o1
        error_imag = w_imag - qw_imag_o1
        qw_real_o2, qw_imag_o2 = PhaseQuantSTE.apply(error_real, error_imag)
        qw_real = qw_real_o1 + qw_real_o2
        qw_imag = qw_imag_o1 + qw_imag_o2
        return qw_real, qw_imag

    @staticmethod
    def backward(ctx, grad_real, grad_imag):
        return grad_real, grad_imag


class PhaseQuantSTE_V3(torch.autograd.Function):
    """Three-step residual quantization"""

    @staticmethod
    def forward(ctx, w_real: torch.Tensor, w_imag: torch.Tensor):
        qw_real_o1, qw_imag_o1 = PhaseQuantSTE.apply(w_real, w_imag)
        error_real_1 = w_real - qw_real_o1
        error_imag_1 = w_imag - qw_imag_o1
        qw_real_o2, qw_imag_o2 = PhaseQuantSTE.apply(error_real_1, error_imag_1)
        error_real_2 = error_real_1 - qw_real_o2
        error_imag_2 = error_imag_1 - qw_imag_o2
        qw_real_o3, qw_imag_o3 = PhaseQuantSTE.apply(error_real_2, error_imag_2)
        qw_real = qw_real_o1 + qw_real_o2 + qw_real_o3
        qw_imag = qw_imag_o1 + qw_imag_o2 + qw_imag_o3
        return qw_real, qw_imag

    @staticmethod
    def backward(ctx, grad_real, grad_imag):
        return grad_real, grad_imag


class PhaseQuantSTE_V4(torch.autograd.Function):
    """Four-step residual quantization"""

    @staticmethod
    def forward(ctx, w_real: torch.Tensor, w_imag: torch.Tensor):
        qw_real_o1, qw_imag_o1 = PhaseQuantSTE.apply(w_real, w_imag)
        error_real_1 = w_real - qw_real_o1
        error_imag_1 = w_imag - qw_imag_o1
        qw_real_o2, qw_imag_o2 = PhaseQuantSTE.apply(error_real_1, error_imag_1)
        error_real_2 = error_real_1 - qw_real_o2
        error_imag_2 = error_imag_1 - qw_imag_o2
        qw_real_o3, qw_imag_o3 = PhaseQuantSTE.apply(error_real_2, error_imag_2)
        error_real_3 = error_real_2 - qw_real_o3
        error_imag_3 = error_imag_2 - qw_imag_o3
        qw_real_o4, qw_imag_o4 = PhaseQuantSTE.apply(error_real_3, error_imag_3)
        qw_real = qw_real_o1 + qw_real_o2 + qw_real_o3 + qw_real_o4
        qw_imag = qw_imag_o1 + qw_imag_o2 + qw_imag_o3 + qw_imag_o4
        return qw_real, qw_imag

    @staticmethod
    def backward(ctx, grad_real, grad_imag):
        return grad_real, grad_imag

import torch
import torch.nn as nn
import torch.nn.functional as F
from quantization import BitNetQuantSTE, PhaseQuantSTE, PhaseQuantSTE_V2, PhaseQuantSTE_V3, PhaseQuantSTE_V4
import math

class QATLinearBitNet(nn.Linear):
    """BitNet QAT linear layer"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        quantized_weight = BitNetQuantSTE.apply(self.weight)
        return F.linear(x, quantized_weight, self.bias)

class QATLinearComplexPhaseV1(nn.Linear):
    """Complex-Phase V1 QAT linear layer"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.in_features % 2 != 0 or self.out_features % 2 != 0:
            raise ValueError("Complex-Phase QAT requires even in/out features for Linear layers.")

    def forward(self, x):
        A = self.weight
        n, m = A.shape[0] // 2, A.shape[1] // 2
        A11, A12 = A[:n, :m], A[:n, m:]
        A21, A22 = A[n:, :m], A[n:, m:]
        
        U_re = 0.5 * (A11 + A22)
        U_im = 0.5 * (A21 - A12)
        W_re = 0.5 * (A11 - A22)
        W_im = 0.5 * (A12 + A21)
        
        U_re_q, U_im_q = PhaseQuantSTE.apply(U_re, U_im)
        W_re_q, W_im_q = PhaseQuantSTE.apply(W_re, W_im)
        
        A11_q = W_re_q + U_re_q
        A12_q = W_im_q - U_im_q
        A21_q = W_im_q + U_im_q
        A22_q = -W_re_q + U_re_q
        
        A_quant_top = torch.cat([A11_q, A12_q], dim=1)
        A_quant_bottom = torch.cat([A21_q, A22_q], dim=1)
        A_quant = torch.cat([A_quant_top, A_quant_bottom], dim=0)

        return F.linear(x, A_quant, self.bias)

class QATLinearComplexPhaseV2(nn.Linear):
    """Complex-Phase V2 QAT linear layer (1-step residual)"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.in_features % 2 != 0 or self.out_features % 2 != 0:
            raise ValueError("Complex-Phase QAT requires even in/out features for Linear layers.")

    def forward(self, x):
        A = self.weight
        n, m = A.shape[0] // 2, A.shape[1] // 2
        A11, A12 = A[:n, :m], A[:n, m:]
        A21, A22 = A[n:, :m], A[n:, m:]
        
        U_re = 0.5 * (A11 + A22)
        U_im = 0.5 * (A21 - A12)
        W_re = 0.5 * (A11 - A22)
        W_im = 0.5 * (A12 + A21)
        
        U_re_q, U_im_q = PhaseQuantSTE_V2.apply(U_re, U_im)
        W_re_q, W_im_q = PhaseQuantSTE_V2.apply(W_re, W_im)
        
        A11_q = W_re_q + U_re_q
        A12_q = W_im_q - U_im_q
        A21_q = W_im_q + U_im_q
        A22_q = -W_re_q + U_re_q
        
        A_quant_top = torch.cat([A11_q, A12_q], dim=1)
        A_quant_bottom = torch.cat([A21_q, A22_q], dim=1)
        A_quant = torch.cat([A_quant_top, A_quant_bottom], dim=0)

        return F.linear(x, A_quant, self.bias)

class QATLinearComplexPhaseV3(nn.Linear):
    """Complex-Phase V3 QAT linear layer (2-step residual)"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.in_features % 2 != 0 or self.out_features % 2 != 0:
            raise ValueError("Complex-Phase QAT requires even in/out features for Linear layers.")

    def forward(self, x):
        A = self.weight
        n, m = A.shape[0] // 2, A.shape[1] // 2
        A11, A12 = A[:n, :m], A[:n, m:]
        A21, A22 = A[n:, :m], A[n:, m:]
        
        U_re = 0.5 * (A11 + A22)
        U_im = 0.5 * (A21 - A12)
        W_re = 0.5 * (A11 - A22)
        W_im = 0.5 * (A12 + A21)
        
        U_re_q, U_im_q = PhaseQuantSTE_V3.apply(U_re, U_im)
        W_re_q, W_im_q = PhaseQuantSTE_V3.apply(W_re, W_im)
        
        A11_q = W_re_q + U_re_q
        A12_q = W_im_q - U_im_q
        A21_q = W_im_q + U_im_q
        A22_q = -W_re_q + U_re_q
        
        A_quant_top = torch.cat([A11_q, A12_q], dim=1)
        A_quant_bottom = torch.cat([A21_q, A22_q], dim=1)
        A_quant = torch.cat([A_quant_top, A_quant_bottom], dim=0)

        return F.linear(x, A_quant, self.bias)

class QATLinearComplexPhaseV4(nn.Linear):
    """Complex-Phase V4 QAT linear layer (3-step residual)"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.in_features % 2 != 0 or self.out_features % 2 != 0:
            raise ValueError("Complex-Phase QAT requires even in/out features for Linear layers.")

    def forward(self, x):
        A = self.weight
        n, m = A.shape[0] // 2, A.shape[1] // 2
        A11, A12 = A[:n, :m], A[:n, m:]
        A21, A22 = A[n:, :m], A[n:, m:]
        
        U_re = 0.5 * (A11 + A22)
        U_im = 0.5 * (A21 - A12)
        W_re = 0.5 * (A11 - A22)
        W_im = 0.5 * (A12 + A21)
        
        U_re_q, U_im_q = PhaseQuantSTE_V4.apply(U_re, U_im)
        W_re_q, W_im_q = PhaseQuantSTE_V4.apply(W_re, W_im)
        
        A11_q = W_re_q + U_re_q
        A12_q = W_im_q - U_im_q
        A21_q = W_im_q + U_im_q
        A22_q = -W_re_q + U_re_q
        
        A_quant_top = torch.cat([A11_q, A12_q], dim=1)
        A_quant_bottom = torch.cat([A21_q, A22_q], dim=1)
        A_quant = torch.cat([A_quant_top, A_quant_bottom], dim=0)

        return F.linear(x, A_quant, self.bias)

METHOD_MAP = {
    'bitnet': QATLinearBitNet,
    'complex_phase_v1': QATLinearComplexPhaseV1,
    'complex_phase_v2': QATLinearComplexPhaseV2,
    'complex_phase_v3': QATLinearComplexPhaseV3,
    'complex_phase_v4': QATLinearComplexPhaseV4,
}

def replace_modules_for_qat(model: nn.Module, method: str, skip_lm_head: bool = False):
    """Recursively replace nn.Linear layers in the model with QAT layers"""
    if method not in METHOD_MAP:
        raise ValueError(f"Unknown method: {method}. Available methods: {list(METHOD_MAP.keys())}")

    TargetQATClass = METHOD_MAP[method]
    
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_modules_for_qat(module, method, skip_lm_head)
        
        if isinstance(module, nn.Linear):
            if skip_lm_head and name == 'lm_head':
                print(f"  -> Skipping lm_head layer (skip_lm_head=True)")
                continue
            
            if 'complex_phase' in method:
                if module.in_features % 2 != 0 or module.out_features % 2 != 0:
                    print(f"  -> Skipping Complex-Phase replacement (non-even dimensions): {name} ({module.in_features}, {module.out_features})")
                    continue
            
            print(f"  -> Replacing layer: {name} with {TargetQATClass.__name__}")
            new_module = TargetQATClass(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                dtype=module.weight.dtype,
                device=module.weight.device
            )
            new_module.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                new_module.bias.data.copy_(module.bias.data)
            
            setattr(model, name, new_module)

class InferenceOptimizedBitNet(nn.Linear):
    """Inference-optimized BitNet linear layer, in-place weight replacement to save memory"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_quantized = False
    
    def _ensure_quantized(self):
        """Ensure weights are quantized, executed only once"""
        if not self._is_quantized:
            with torch.no_grad():
                w = self.weight
                scale = w.abs().mean()
                alpha = w.mean()
                centered_w = w - alpha
                binarized_w = torch.where(centered_w > 0, 1.0, -1.0).to(w.dtype)
                quantized_w = binarized_w * scale
                self.weight.data = quantized_w
                self._is_quantized = True
    
    def forward(self, x):
        self._ensure_quantized()
        return F.linear(x, self.weight, self.bias)

class InferenceOptimizedComplexPhase(nn.Linear):
    """Inference-optimized Complex Phase linear layer, supports V1-V4"""
    def __init__(self, version="v1", *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.in_features % 2 != 0 or self.out_features % 2 != 0:
            raise ValueError("Complex-Phase requires even in/out features.")
        self._is_quantized = False
        self._version = version.lower()
        if self._version not in ["v1", "v2", "v3", "v4"]:
            raise ValueError(f"Unsupported version: {version}. Must be one of ['v1', 'v2', 'v3', 'v4']")
    
    def _ensure_quantized(self):
        """Ensure weights are quantized, executed only once"""
        if not self._is_quantized:
            with torch.no_grad():
                A = self.weight
                n, m = A.shape[0] // 2, A.shape[1] // 2
                A11, A12 = A[:n, :m], A[:n, m:]
                A21, A22 = A[n:, :m], A[n:, m:]
                
                U_re = 0.5 * (A11 + A22)
                U_im = 0.5 * (A21 - A12)
                W_re = 0.5 * (A11 - A22)
                W_im = 0.5 * (A12 + A21)
                
                if self._version == "v1":
                    U_re_q, U_im_q = self._phase_quant_v1(U_re, U_im)
                    W_re_q, W_im_q = self._phase_quant_v1(W_re, W_im)
                elif self._version == "v2":
                    U_re_q, U_im_q = self._phase_quant_v2(U_re, U_im)
                    W_re_q, W_im_q = self._phase_quant_v2(W_re, W_im)
                elif self._version == "v3":
                    U_re_q, U_im_q = self._phase_quant_v3(U_re, U_im)
                    W_re_q, W_im_q = self._phase_quant_v3(W_re, W_im)
                elif self._version == "v4":
                    U_re_q, U_im_q = self._phase_quant_v4(U_re, U_im)
                    W_re_q, W_im_q = self._phase_quant_v4(W_re, W_im)
                
                A11_q = W_re_q + U_re_q
                A12_q = W_im_q - U_im_q
                A21_q = W_im_q + U_im_q
                A22_q = -W_re_q + U_re_q
                
                A_quant_top = torch.cat([A11_q, A12_q], dim=1)
                A_quant_bottom = torch.cat([A21_q, A22_q], dim=1)
                A_quant = torch.cat([A_quant_top, A_quant_bottom], dim=0)
                
                self.weight.data = A_quant
                self._is_quantized = True
    
    def _phase_quant_v1(self, w_real, w_imag):
        """V1: Basic PhaseQuant"""
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
        
        return qw_real * s_re, qw_imag * s_im
    
    def _phase_quant_v2(self, w_real, w_imag):
        """V2: 1-step residual quantization"""
        qw_real_o1, qw_imag_o1 = self._phase_quant_v1(w_real, w_imag)
        error_real = w_real - qw_real_o1
        error_imag = w_imag - qw_imag_o1
        qw_real_o2, qw_imag_o2 = self._phase_quant_v1(error_real, error_imag)
        qw_real = qw_real_o1 + qw_real_o2
        qw_imag = qw_imag_o1 + qw_imag_o2
        return qw_real, qw_imag
    
    def _phase_quant_v3(self, w_real, w_imag):
        """V3: 2-step residual quantization"""
        qw_real_o1, qw_imag_o1 = self._phase_quant_v1(w_real, w_imag)
        error_real_1 = w_real - qw_real_o1
        error_imag_1 = w_imag - qw_imag_o1
        qw_real_o2, qw_imag_o2 = self._phase_quant_v1(error_real_1, error_imag_1)
        error_real_2 = error_real_1 - qw_real_o2
        error_imag_2 = error_imag_1 - qw_imag_o2
        qw_real_o3, qw_imag_o3 = self._phase_quant_v1(error_real_2, error_imag_2)
        qw_real = qw_real_o1 + qw_real_o2 + qw_real_o3
        qw_imag = qw_imag_o1 + qw_imag_o2 + qw_imag_o3
        return qw_real, qw_imag
    
    def _phase_quant_v4(self, w_real, w_imag):
        """V4: 3-step residual quantization"""
        qw_real_o1, qw_imag_o1 = self._phase_quant_v1(w_real, w_imag)
        error_real_1 = w_real - qw_real_o1
        error_imag_1 = w_imag - qw_imag_o1
        qw_real_o2, qw_imag_o2 = self._phase_quant_v1(error_real_1, error_imag_1)
        error_real_2 = error_real_1 - qw_real_o2
        error_imag_2 = error_imag_1 - qw_imag_o2
        qw_real_o3, qw_imag_o3 = self._phase_quant_v1(error_real_2, error_imag_2)
        error_real_3 = error_real_2 - qw_real_o3
        error_imag_3 = error_imag_2 - qw_imag_o3
        qw_real_o4, qw_imag_o4 = self._phase_quant_v1(error_real_3, error_imag_3)
        qw_real = qw_real_o1 + qw_real_o2 + qw_real_o3 + qw_real_o4
        qw_imag = qw_imag_o1 + qw_imag_o2 + qw_imag_o3 + qw_imag_o4
        return qw_real, qw_imag
    
    def forward(self, x):
        self._ensure_quantized()
        return F.linear(x, self.weight, self.bias)

def convert_to_inference_mode(model):
    """Convert QAT modules to inference-optimized version (permanently modifies model weights)"""
    converted_count = 0
    
    def _convert_module(module, name_path=""):
        nonlocal converted_count
        
        for name, child in list(module.named_children()):
            full_name = f"{name_path}.{name}" if name_path else name
            
            if isinstance(child, QATLinearBitNet):
                new_module = InferenceOptimizedBitNet(
                    child.in_features, 
                    child.out_features, 
                    bias=child.bias is not None,
                    device=child.weight.device,
                    dtype=child.weight.dtype
                )
                new_module.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    new_module.bias.data.copy_(child.bias.data)
                
                setattr(module, name, new_module)
                converted_count += 1
                print(f"  -> Converting BitNet layer: {full_name}")
                
            elif isinstance(child, (QATLinearComplexPhaseV1, QATLinearComplexPhaseV2, 
                                   QATLinearComplexPhaseV3, QATLinearComplexPhaseV4)):
                if isinstance(child, QATLinearComplexPhaseV1):
                    version = "v1"
                elif isinstance(child, QATLinearComplexPhaseV2):
                    version = "v2"
                elif isinstance(child, QATLinearComplexPhaseV3):
                    version = "v3"
                elif isinstance(child, QATLinearComplexPhaseV4):
                    version = "v4"
                
                new_module = InferenceOptimizedComplexPhase(
                    version=version,
                    in_features=child.in_features, 
                    out_features=child.out_features, 
                    bias=child.bias is not None,
                    device=child.weight.device,
                    dtype=child.weight.dtype
                )
                new_module.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    new_module.bias.data.copy_(child.bias.data)
                
                setattr(module, name, new_module)
                converted_count += 1
                print(f"  -> Converting ComplexPhase{version.upper()} layer: {full_name}")
            else:
                _convert_module(child, full_name)
    
    _convert_module(model)
    print(f"Converted {converted_count} QAT layers to inference-optimized version")
    return model
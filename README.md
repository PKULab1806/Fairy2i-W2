# Fairy2i-W2

## Table of Contents

- [📝 Abstract](#-abstract)
- [🔬 Method](#-method)
- [📊 Evaluation](#-evaluation)
- [🚀 Quick Start](#-quick-start)
- [📁 Repository Structure](#-repository-structure)
- [📚 Citation](#-citation)
- [⚖️ License](#️-license)
- [📧 Contact](#-contact)

## 📝 Abstract

Large language models (LLMs) have revolutionized artificial intelligence, yet their massive memory and computational demands necessitate aggressive quantization, increasingly pushing representations toward the theoretical limit of a single bit. While complex-valued LLMs, such as iFairy, offer a superior chance for low-bit representation compared to real-valued counterparts, they require training from scratch, preventing the utilization of the vast ecosystem of pre-trained real-valued foundation models.

Here we present **Fairy2i**, a universal framework that transforms pre-trained real-valued layers into an equivalent widely-linear complex form, enabling extremely low-bit quantization while reusing existing checkpoints. By proving a lossless mathematical equivalence between real and widely-linear maps, we convert standard Transformers into the complex domain and employ a phase-aware quantization scheme with a highly efficient codebook of fourth roots of unity $\{\pm 1, \pm i\}$. Furthermore, we introduce a recursive residual quantization mechanism that iteratively minimizes quantization error, allowing inference to proceed via efficient multiplication-free accumulation.

We demonstrate that **Fairy2i-W2** restores the performance of LLaMA-2 7B at an effective 2-bit precision to levels nearly comparable with full-precision baselines, significantly outperforming state-of-the-art real-valued binary and ternary quantization methods.

This work bridges the gap between the representational efficiency of complex-valued arithmetic and the practical utility of pre-trained models, paving a new way for efficient inference on commodity hardware.

## 🔬 Method

Fairy2i-W2 consists of three key components:

### 🔄 Widely-Linear Transformation

We transform pre-trained real-valued linear layers into an equivalent **widely-linear complex form** without altering the model's behavior. Each real linear layer $R$ (a real matrix of size $2n \times 2m$) is reparameterized into two complex matrices $U$ and $W$ (each of size $n \times m$) such that:

$$y = Ux + W\bar{x}$$

where $\bar{x}$ denotes the complex conjugate of $x$. This transformation is **lossless** and **unique**, preserving the original forward computation before quantization.

### ⚡ Phase-Aware Complex Quantization

We quantize complex weights using a phase-based scheme with the codebook $\{\pm 1, \pm i\}$ (fourth roots of unity). For each complex weight, we project it to the nearest codeword by angle and apply axis-wise scaling factors. During QAT training, we maintain full-precision master weights and use quantized copies in the forward pass with straight-through estimator (STE) gradients.

### 🔁 Recursive Residual Quantization

To further reduce quantization error, we recursively quantize the residual error. Each complex weight is represented as a sum of low-bit terms:

$$W_q \approx \sum_{t=0}^{T-1} W^{(t)}$$

where each term is quantized using the same phase-aware mechanism. For **Fairy2i-W2** ($T=2$), we use 2 recursive stages, achieving an effective **2 bits per real parameter**.

## Evaluation

### 📈 Main Results on LLaMA-2 7B

| Method | Bits | C4 PPL↓ | ARC-e | ARC-c | HellaSwag | PIQA | Winogrande | Avg. |
|--------|------|---------|-------|-------|-----------|------|------------|------|
| LLaMA-2 (FP16) | 16 | 6.63 | 75.59 | 43.17 | 57.06 | 77.91 | 69.85 | 64.72 |
| **Fairy2i-W2** | **2** | **7.85** | **72.73** | **39.76** | **53.33** | **76.17** | **68.03** | **62.00** |
| AQLM | 2 | 8.54 | 63.68 | 32.76 | 49.55 | 74.76 | 65.67 | 57.28 |
| QuIP# | 2 | 11.01 | 55.56 | 28.84 | 42.94 | 71.38 | 62.43 | 52.23 |
| Real-Ternary (QAT) | 1.58 | 11.06 | 55.93 | 24.15 | 38.43 | 69.80 | 55.17 | 48.70 |
| **Fairy2i-W1** | **1** | **11.03** | **56.56** | **24.82** | **38.19** | **70.08** | **53.67** | **48.66** |
| Real-Binary (QAT) | 1 | 11.75 | 53.32 | 22.70 | 35.57 | 66.81 | 52.64 | 46.21 |
| GPTQ | 3 | 10.61 | 58.46 | 31.06 | 45.21 | 71.49 | 59.19 | 53.08 |

**Key Results:**
- **Fairy2i-W2 (2-bit)** achieves a perplexity of 7.85, closing the gap to FP16 (6.63) while outperforming all 2-bit PTQ methods
- **Fairy2i-W2** achieves 62.00% average accuracy on zero-shot tasks, highly competitive with FP16 (64.72%)
- **Fairy2i-W1 (1-bit)** outperforms real-valued binary and ternary baselines at the same or lower bit budgets

## 🚀 Quick Start

**Fairy2i-W2** is based on LLaMA-2 7B architecture, with only the linear layers replaced by complex-valued QAT layers. The model structure is otherwise identical to LLaMA-2.

### 📦 Installation

```bash
pip install torch transformers safetensors huggingface_hub accelerate datasets lm-eval
```

### 🔄 Loading the Model

The model can be loaded using the `model_module` package. Here's a basic example:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from model_module.qat_modules import replace_modules_for_qat, convert_to_inference_mode
import torch

# Load base model
model_path = "meta-llama/Llama-2-7b-hf"  # or your local path
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Replace linear layers with QAT modules
replace_modules_for_qat(model, "complex_phase_v2", skip_lm_head=False)

# Convert to inference mode for faster inference
convert_to_inference_mode(model)

# The model is ready to use!
prompt = "Hello, how are you?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### 🏋️ Training

To train a model with QAT, use the training script:

```bash
cd train
bash train.sh
```

The training script supports the following arguments:
- `--quant_method`: QAT quantization method (choices: `bitnet`, `complex_phase_v1`, `complex_phase_v2`, `complex_phase_v3`, `complex_phase_v4`)
- `--skip_lm_head`: Whether to skip replacement of lm_head layer (default: False)

### ✅ Evaluation

#### 📉 Perplexity Evaluation

Evaluate perplexity on Wikitext-2 and C4 datasets:

```bash
cd eval
bash eval_ppl.sh
```

#### 🎯 Task Evaluation

Evaluate on downstream tasks using lm-eval:

```bash
cd eval
bash eval_task.sh
```

### ℹ️ Model Details

- **Base Model**: LLaMA-2 7B
- **Quantization Method**: Complex-Phase V2 (2-step recursive residual quantization)
- **Effective Bit Width**: 2 bits per real parameter
- **Codebook**: $\{\pm 1, \pm i\}$ (fourth roots of unity)
- **Training**: QAT (Quantization-Aware Training) on 30B tokens from RedPajama dataset

## 📁 Repository Structure

```
fairy2i-w2-repo-github/
├── README.md
├── model_module/
│   ├── __init__.py
│   ├── qat_modules.py          # QAT linear layer implementations
│   └── quantization.py         # Quantization functions (PhaseQuant, BitNet, etc.)
├── train/
│   ├── train.py                # Training script
│   ├── train.sh                # Training launch script
│   └── complexnet_config.yaml  # Accelerate configuration
└── eval/
    ├── eval_ppl.py             # Perplexity evaluation script
    ├── eval_ppl.sh             # Perplexity evaluation launcher
    ├── eval_task.py            # Task evaluation script
    ├── eval_task.sh            # Task evaluation launcher
    └── eval_utils.py            # Evaluation utilities
```

## 📚 Citation

If you use Fairy2i-W2 in your research, please cite:

```bibtex
@article{wang2025fairy2i,
  title={Fairy2i: Training Complex LLMs from Real LLMs with All Parameters in $\{\pm 1, \pm i\}$},
  author={Wang, Feiyu and Tan, Xinyu and Huang, Bokai and Zhang, Yihao and Wang, Guoan and Cong, Peizhuang and Yang, Tong},
  journal={arXiv preprint},
  year={2025}
}
```

## ⚖️ License

This model follows the same license as LLaMA-2. Please refer to the original LLaMA-2 license for details.

## 📧 Contact

For questions or issues, please contact: yangtong@pku.edu.cn

#!/usr/bin/env python3
"""
Export Qwen2.5 with Transformers 4.37.2 (No vmap issues)
========================================================

This version works with transformers 4.37.2 which doesn't have
the vmap-based causal mask issues.

Usage:
    python3 export_qwen_clean.py --output qwen_clean128.onnx --seq-len 128
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import warnings

warnings.filterwarnings("ignore")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoModelForCausalLM, AutoConfig


class CleanQwenWrapper(nn.Module):
    """
    Wrapper with fixed sequence length (no KV cache).
    Compatible with transformers 4.37.2
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = model.config

    def forward(self, input_ids: torch.Tensor):
        """
        Forward pass with only input_ids.

        Args:
            input_ids: (batch_size, seq_len)

        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape

        # Create attention mask (all 1s for valid tokens)
        attention_mask = torch.ones(batch_size, seq_len, device=input_ids.device, dtype=torch.long)

        # Create position IDs
        position_ids = torch.arange(seq_len, device=input_ids.device, dtype=torch.long)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Call model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
        )

        return outputs[0]  # logits


def export_clean_onnx(
    model_name_or_path: str,
    output_path: str,
    seq_len: int = 128,
    opset: int = 17,
    device: str = "cpu",
):
    """
    Export Qwen2.5 to ONNX with transformers 4.37.2.
    """
    print("=" * 70)
    print("Qwen2.5 Clean ONNX Export (Transformers 4.37.2)")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Model: {model_name_or_path}")
    print(f"  Output: {output_path}")
    print(f"  Sequence length: {seq_len} (FIXED)")
    print(f"  Device: {device}")

    # Load model
    print(f"\n[1/4] Loading model...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )
        model = model.to(device)
        model.eval()

        config = model.config
        print(f"  - Hidden size: {config.hidden_size}")
        print(f"  - Num layers: {config.num_hidden_layers}")
        print(f"  - Num attention heads: {config.num_attention_heads}")
        print(f"  - Num KV heads: {config.num_key_value_heads}")
        print(f"  - Vocab size: {config.vocab_size}")

    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Wrap model
    print(f"\n[2/4] Wrapping model...")
    wrapped_model = CleanQwenWrapper(model)
    wrapped_model.eval()

    # Create dummy inputs
    print(f"\n[3/4] Creating dummy inputs (seq_len={seq_len})...")
    dummy_input_ids = torch.randint(
        0, config.vocab_size,
        (1, seq_len),
        dtype=torch.long,
        device=device
    )

    # Test forward pass first
    print(f"[INFO] Testing forward pass before export...")
    with torch.no_grad():
        test_output = wrapped_model(dummy_input_ids)
        print(f"[OK] Forward pass successful, output shape: {test_output.shape}")

    # Input/Output names
    input_names = ["input_ids"]
    output_names = ["logits"]

    # Set environment
    os.environ["ONNX_EXPORT_USE_MLIR"] = "0"

    # Export - use torch.jit.trace directly to avoid torch.export issues
    print(f"\n[4/4] Exporting to ONNX (using torch.jit.trace)...")

    try:
        with torch.no_grad():
            print(f"[INFO] Tracing model with torch.jit.trace...")
            traced_model = torch.jit.trace(wrapped_model, (dummy_input_ids,))
            traced_model = torch.jit.freeze(traced_model)

            print(f"[INFO] Exporting traced model to ONNX...")
            # Use standard ONNX export for traced models (PyTorch 2.1)
            torch.onnx.export(
                traced_model,
                (dummy_input_ids,),
                output_path,
                opset_version=opset,
                input_names=input_names,
                output_names=output_names,
                export_params=True,
                do_constant_folding=True,
                verbose=False,
            )
        print(f"[OK] Exported to: {output_path}")

    except Exception as e:
        print(f"[ERROR] Export failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Verify
    print(f"\n[Verifying ONNX model...")
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("[OK] ONNX model is valid")

        print(f"\nModel inputs:")
        for inp in onnx_model.graph.input:
            dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
            print(f"  - {inp.name}: shape={tuple(dims)}, dtype={inp.type.tensor_type.elem_type}")

        print(f"\nModel outputs:")
        for out in onnx_model.graph.output:
            dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
            print(f"  - {out.name}: shape={tuple(dims)}, dtype={out.type.tensor_type.elem_type}")

    except ImportError:
        print("[WARNING] onnx not installed, skipping verification")
    except Exception as e:
        print(f"[WARNING] Verification failed: {e}")

    # Save info
    info_path = output_path.replace(".onnx", "_info.txt")
    with open(info_path, "w") as f:
        f.write(f"Qwen2.5 Clean ONNX Export Information\n")
        f.write(f"=" * 50 + "\n\n")
        f.write(f"Model: {model_name_or_path}\n")
        f.write(f"Output: {output_path}\n")
        f.write(f"Sequence length: {seq_len} (FIXED)\n")
        f.write(f"Device: {device}\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  - Hidden size: {config.hidden_size}\n")
        f.write(f"  - Num layers: {config.num_hidden_layers}\n")
        f.write(f"  - Num KV heads: {config.num_key_value_heads}\n")
        f.write(f"  - Vocab size: {config.vocab_size}\n")
        f.write(f"  - Opset version: {opset}\n\n")
        f.write(f"Input shapes:\n")
        f.write(f"  - input_ids: (1, {seq_len}) - FIXED\n")

    print(f"\n[OK] Model info saved to: {info_path}")
    print("\n" + "=" * 70)
    print("Export completed successfully!")
    print("=" * 70)
    print(f"\nNext steps:")
    print(f"  1. Convert to OM: ./convert_fixed.sh --onnx-model {output_path} --output qwen_clean{seq_len}")
    print(f"  2. Run inference: python3 acl_inference_patched.py --model qwen_clean{seq_len}.om")


def main():
    parser = argparse.ArgumentParser(
        description="Export Qwen2.5 with transformers 4.37.2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export with seq_len=128
  python3 export_qwen_clean.py --output qwen_clean128.onnx --seq-len 128

  # Export with seq_len=512
  python3 export_qwen_clean.py --output qwen_clean512.onnx --seq-len 512
        """
    )

    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Model name or path (default: Qwen/Qwen2.5-0.5B-Instruct)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="qwen2.5-0.5b-clean128.onnx",
        help="Output ONNX file path",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=128,
        help="Fixed sequence length (default: 128)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Target device (default: cpu)",
    )

    args = parser.parse_args()

    export_clean_onnx(
        model_name_or_path=args.model,
        output_path=args.output,
        seq_len=args.seq_len,
        opset=args.opset,
        device=args.device,
    )


if __name__ == "__main__":
    main()

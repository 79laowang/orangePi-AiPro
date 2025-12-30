#!/usr/bin/env python3
"""
Qwen2.5 Patched ACL Inference
==============================

Simplified inference for models exported with the patched export script.
Only takes input_ids as input (no attention_mask or position_ids).

Usage:
    python3 acl_inference_patched.py --model qwen_patch128.om --prompt "Hello"
"""

import os
import sys
import time
import argparse
import numpy as np

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["ASCEND_LAUNCH_BLOCKING"] = "1"
os.environ["ASCEND_GLOBAL_LOG_LEVEL"] = "1"
os.environ["LOGLEVEL"] = "WARNING"

try:
    import acl
    import acl.mdl as aclmdl
    import acl.rt as aclrt
except ImportError:
    print("[ERROR] ACL module not found. Source CANN environment first.")
    sys.exit(1)

# Model configuration
MODEL_CONFIG = {
    "vocab_size": 151936,
    "eos_token_id": 151645,
}


class TokenizerWrapper:
    """Tokenizer wrapper with HF mirror support."""

    def __init__(self, model_name_or_path: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        if os.getenv("HF_ENDPOINT") is None:
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path, trust_remote_code=True,
            )
            print(f"[OK] Tokenizer loaded from: {model_name_or_path}")
        except Exception as e:
            print(f"[WARNING] Failed to load tokenizer: {e}")
            self.tokenizer = None

    def encode(self, text: str) -> np.ndarray:
        if self.tokenizer:
            tokens = self.tokenizer.encode(text, return_tensors="np")
            return tokens[0].astype(np.int64)
        else:
            return np.array([ord(c) % MODEL_CONFIG["vocab_size"] for c in text], dtype=np.int64)

    def decode(self, token_ids: np.ndarray) -> str:
        if self.tokenizer:
            return self.tokenizer.decode(token_ids, skip_special_tokens=False)
        else:
            return "".join(chr(int(tid)) if tid < 128 else f"[{tid}]" for tid in token_ids)


class PatchedQwenInference:
    """Inference for patched models (only input_ids)."""

    def __init__(self, model_path: str, device_id: int = 0):
        self.model_path = model_path
        self.device_id = device_id
        self.context = None
        self.model_id = None
        self.model_desc = None
        self.dataset_input = None
        self.dataset_output = None
        self._initialized = False

        # Input/output info
        self.input_size = None
        self.input_shape = None
        self.output_size = None
        self.output_shape = None
        self.seq_len = 128  # Default, will be detected

    def init(self):
        """Initialize ACL and load model."""
        print("=" * 70)
        print("Qwen2.5 Patched ACL Inference")
        print("=" * 70)

        # Initialize ACL
        ret = acl.init()
        if ret != 0:
            raise RuntimeError(f"acl.init failed: {ret}")

        # Set device
        ret = aclrt.set_device(self.device_id)
        if ret != 0:
            raise RuntimeError(f"aclrt.set_device failed: {ret}")

        # Create context
        self.context, ret = aclrt.create_context(self.device_id)
        if ret != 0:
            raise RuntimeError(f"aclrt.create_context failed: {ret}")

        # Load model
        print(f"\n[1/4] Loading model: {self.model_path}")
        self.model_id, ret = aclmdl.load_from_file(self.model_path)
        if ret != 0:
            raise RuntimeError(f"aclmdl.load_from_file failed: {ret}")
        print(f"[OK] Model loaded, ID: {self.model_id}")

        # Get model description
        self.model_desc = aclmdl.create_desc()
        ret = aclmdl.get_desc(self.model_desc, self.model_id)
        if ret != 0:
            raise RuntimeError(f"aclmdl.get_desc failed: {ret}")

        # Get model info
        self._get_model_info()

        # Create datasets
        self._create_dataset()

        self._initialized = True
        print("\n[OK] ACL initialized successfully")

    def _get_model_info(self):
        """Get model information."""
        print(f"\n[2/4] Getting model info...")

        num_inputs = aclmdl.get_num_inputs(self.model_desc)
        num_outputs = aclmdl.get_num_outputs(self.model_desc)

        print(f"  Inputs:  {num_inputs}")
        print(f"  Outputs: {num_outputs}")

        # Get input info
        for i in range(num_inputs):
            name = aclmdl.get_input_name_by_index(self.model_desc, i)
            self.input_size = aclmdl.get_input_size_by_index(self.model_desc, i)
            dims_result = aclmdl.get_input_dims_v2(self.model_desc, i)
            dtype = aclmdl.get_input_data_type(self.model_desc, i)

            if dims_result and len(dims_result) > 0:
                dims = dims_result[0].get('dims', [])
            else:
                dims = []

            self.input_shape = dims

            # Detect sequence length
            if len(dims) >= 2:
                self.seq_len = dims[1] if dims[1] > 0 else 128

            print(f"  Input[{i}] {name}: shape={tuple(dims)}, dtype={dtype}, size={self.input_size}")

        # Get output info
        for i in range(num_outputs):
            name = aclmdl.get_output_name_by_index(self.model_desc, i)
            self.output_size = aclmdl.get_output_size_by_index(self.model_desc, i)
            dims_result = aclmdl.get_output_dims(self.model_desc, i)
            dtype = aclmdl.get_output_data_type(self.model_desc, i)

            if dims_result and len(dims_result) > 0:
                dims = dims_result[0].get('dims', [])
            else:
                dims = []

            self.output_shape = dims
            print(f"  Output[{i}] {name}: shape={tuple(dims)}, dtype={dtype}, size={self.output_size}")

        print(f"\n[INFO] Sequence length: {self.seq_len}")

    def _create_dataset(self):
        """Create input/output datasets."""
        print(f"\n[3/4] Creating datasets...")

        # Input dataset
        self.dataset_input = aclmdl.create_dataset()
        data, ret = aclrt.malloc(self.input_size, 0)
        if ret != 0:
            raise RuntimeError(f"aclrt.malloc failed for input")
        data_buffer = acl.create_data_buffer(data, self.input_size)
        result = aclmdl.add_dataset_buffer(self.dataset_input, data_buffer)
        ret = result[1] if isinstance(result, tuple) and len(result) > 1 else result
        if ret != 0:
            raise RuntimeError(f"aclmdl.add_dataset_buffer failed for input")

        # Output dataset
        self.dataset_output = aclmdl.create_dataset()
        data, ret = aclrt.malloc(self.output_size, 0)
        if ret != 0:
            raise RuntimeError(f"aclrt.malloc failed for output")
        data_buffer = acl.create_data_buffer(data, self.output_size)
        result = aclmdl.add_dataset_buffer(self.dataset_output, data_buffer)
        ret = result[1] if isinstance(result, tuple) and len(result) > 1 else result
        if ret != 0:
            raise RuntimeError(f"aclmdl.add_dataset_buffer failed for output")

        print(f"[OK] Datasets created")

    def forward(self, input_ids: np.ndarray) -> np.ndarray:
        """
        Run forward pass.

        Args:
            input_ids: Token IDs (seq_len,)

        Returns:
            Logits: (1, seq_len, vocab_size)
        """
        actual_seq_len = len(input_ids)

        # Pad to model's sequence length (right-pad with zeros)
        if actual_seq_len > self.seq_len:
            input_ids = input_ids[-self.seq_len:]
            actual_seq_len = self.seq_len

        # Right-pad with zeros - positions 0..actual_seq_len-1 have real tokens
        padded_ids = np.zeros(self.seq_len, dtype=np.int64)
        padded_ids[:actual_seq_len] = input_ids

        # Reshape to (1, seq_len)
        input_batch = padded_ids.reshape(1, -1)

        # Get input buffer
        buffer = aclmdl.get_dataset_buffer(self.dataset_input, 0)
        ptr = acl.get_data_buffer_addr(buffer)
        size = acl.get_data_buffer_size(buffer)

        # Copy input to device
        ret = aclrt.memcpy(ptr, size, input_batch.ctypes.data, input_batch.nbytes, 1)
        if ret != 0:
            raise RuntimeError(f"aclrt.memcpy failed for input: {ret}")

        # Execute
        ret = aclmdl.execute(self.model_id, self.dataset_input, self.dataset_output)
        if ret != 0:
            raise RuntimeError(f"aclmdl.execute failed: {ret}")

        # Get output
        buffer = aclmdl.get_dataset_buffer(self.dataset_output, 0)
        ptr = acl.get_data_buffer_addr(buffer)
        size = acl.get_data_buffer_size(buffer)

        # Copy output from device
        num_elements = size // 4  # float32
        logits = np.zeros(num_elements, dtype=np.float32)
        ret = aclrt.memcpy(logits.ctypes.data, size, ptr, size, 2)
        if ret != 0:
            raise RuntimeError(f"aclrt.memcpy failed for output: {ret}")

        # Reshape to (1, seq_len, vocab_size)
        if self.output_shape:
            shape = tuple(1 if d == -1 else d for d in self.output_shape)
            logits = logits.reshape(shape)

        return logits, actual_seq_len

    def generate(self, prompt: str, tokenizer: TokenizerWrapper,
                 max_tokens: int = 50, callback=None) -> str:
        """Generate text autoregressively."""
        if not self._initialized:
            raise RuntimeError("ACL not initialized")

        print(f"\n[4/4] Generating text...")
        print(f"{'=' * 70}")
        print(f"Prompt: {prompt}")
        print(f"Max tokens: {max_tokens}")
        print(f"{'=' * 70}")

        # Tokenize
        input_ids = tokenizer.encode(prompt)
        print(f"[INFO] Prompt tokens: {len(input_ids)}")

        if len(input_ids) > self.seq_len:
            input_ids = input_ids[-self.seq_len + max_tokens:]
            print(f"[WARNING] Prompt truncated to {len(input_ids)} tokens")

        start_time = time.time()
        generated_tokens = []

        for step in range(max_tokens):
            step_start = time.time()

            # Forward pass
            logits, actual_seq_len = self.forward(input_ids)

            # Get last position logits: (1, seq_len, vocab_size) -> (vocab_size,)
            # Use actual_seq_len - 1 as the position (0-indexed)
            # But be careful: when actual_seq_len == seq_len, use -1
            if actual_seq_len == self.seq_len:
                last_logits = logits[0, -1, :] if len(logits.shape) == 3 else logits[-1]
            else:
                last_logits = logits[0, actual_seq_len - 1, :] if len(logits.shape) == 3 else logits[actual_seq_len - 1]

            # Sample (greedy)
            next_token = int(np.argmax(last_logits))
            generated_tokens.append(next_token)

            if callback:
                callback(next_token)

            # Check EOS
            if next_token == MODEL_CONFIG["eos_token_id"]:
                print(f"\n[INFO] EOS token reached")
                break

            # Append and truncate if needed
            input_ids = np.append(input_ids, next_token)
            if len(input_ids) > self.seq_len:
                input_ids = input_ids[-self.seq_len:]

            step_time = time.time() - step_start
            if step < 3 or step % 10 == 0:
                print(f"[Step {step+1}/{max_tokens}] Token: {next_token}, {step_time*1000:.1f}ms")

        total_time = time.time() - start_time
        print(f"\n[OK] Generated {len(generated_tokens)} tokens in {total_time:.2f}s ({len(generated_tokens)/total_time:.2f} tok/s)")

        return tokenizer.decode(np.array(generated_tokens))

    def finalize(self):
        """Release resources."""
        print(f"\n[Cleanup] Releasing resources...")

        if self.dataset_input is not None:
            for i in range(aclmdl.get_dataset_num_buffers(self.dataset_input)):
                buffer = aclmdl.get_dataset_buffer(self.dataset_input, i)
                data_ptr = acl.get_data_buffer_addr(buffer)
                if data_ptr:
                    aclrt.free(data_ptr)
                acl.destroy_data_buffer(buffer)
            aclmdl.destroy_dataset(self.dataset_input)

        if self.dataset_output is not None:
            for i in range(aclmdl.get_dataset_num_buffers(self.dataset_output)):
                buffer = aclmdl.get_dataset_buffer(self.dataset_output, i)
                data_ptr = acl.get_data_buffer_addr(buffer)
                if data_ptr:
                    aclrt.free(data_ptr)
                acl.destroy_data_buffer(buffer)
            aclmdl.destroy_dataset(self.dataset_output)

        if self.model_desc is not None:
            aclmdl.destroy_desc(self.model_desc)
        if self.model_id is not None:
            aclmdl.unload(self.model_id)
        if self.context is not None:
            aclrt.destroy_context(self.context)

        aclrt.reset_device(self.device_id)
        acl.finalize()
        print("[OK] Resources released")


def main():
    parser = argparse.ArgumentParser(description="Qwen2.5 Patched ACL Inference")
    parser.add_argument("--model", type=str, required=True, help="Path to .om model file")
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--prompt", type=str, default="Hello, how are you?")
    parser.add_argument("--max-tokens", type=int, default=30)
    parser.add_argument("--device", type=int, default=0)

    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"[ERROR] Model not found: {args.model}")
        sys.exit(1)

    tokenizer = TokenizerWrapper(args.tokenizer)
    inference = PatchedQwenInference(args.model, args.device)

    try:
        inference.init()

        def stream_callback(tid):
            text = tokenizer.decode(np.array([tid]))
            print(text, end="", flush=True)

        result = inference.generate(
            prompt=args.prompt,
            tokenizer=tokenizer,
            max_tokens=args.max_tokens,
            callback=stream_callback,
        )

        print(f"\n\n{'=' * 70}")
        print(f"Output:")
        print(f"{'=' * 70}")
        print(result)

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        inference.finalize()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Test vLLM-Ascend with Qwen model
Run this script to verify vLLM-Ascend works on your Ascend 310B4
"""

import sys
import time

def test_vllm():
    """Test vLLM with simple prompts."""
    print("=" * 60)
    print("vLLM-Ascend Test on Ascend 310B4")
    print("=" * 60)

    # Test 1: Import vLLM
    print("\n[Test 1/4] Importing vLLM...")
    try:
        from vllm import LLM, SamplingParams
        print("✅ vLLM imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import vLLM: {e}")
        print("\nPlease install vLLM-Ascend:")
        print("  pip install vllm-ascend")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

    # Test 2: Initialize model
    print("\n[Test 2/4] Initializing model...")
    print("Model: Qwen/Qwen2.5-0.5B-Instruct")
    print("Note: First run will download model (~1GB, may take 5-10 minutes)")

    try:
        llm = LLM(
            model="Qwen/Qwen2.5-0.5B-Instruct",
            max_model_len=2048,
            trust_remote_code=True
        )
        print("✅ Model initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize model: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 3: Simple generation
    print("\n[Test 3/4] Testing simple generation...")
    prompts = ["Hello, my name is"]
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=50
    )

    try:
        start = time.time()
        outputs = llm.generate(prompts, sampling_params)
        elapsed = time.time() - start

        generated_text = outputs[0].outputs[0].text
        words = len(generated_text.split())
        tokens_per_sec = words / elapsed if elapsed > 0 else 0

        print(f"✅ Generation successful!")
        print(f"   Prompt: {prompts[0]}")
        print(f"   Generated: {generated_text}")
        print(f"   Time: {elapsed:.2f}s")
        print(f"   Words: {words}")
        print(f"   Speed: {tokens_per_sec:.2f} words/s")

    except Exception as e:
        print(f"❌ Failed to generate: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 4: Chinese generation
    print("\n[Test 4/4] Testing Chinese generation...")
    prompts = ["北京是中国的"]
    sampling_params = SamplingParams(
        temperature=0.1,
        max_tokens=50
    )

    try:
        start = time.time()
        outputs = llm.generate(prompts, sampling_params)
        elapsed = time.time() - start

        generated_text = outputs[0].outputs[0].text
        chars = len(generated_text)
        chars_per_sec = chars / elapsed if elapsed > 0 else 0

        print(f"✅ Chinese generation successful!")
        print(f"   Prompt: {prompts[0]}")
        print(f"   Generated: {generated_text}")
        print(f"   Time: {elapsed:.2f}s")
        print(f"   Characters: {chars}")
        print(f"   Speed: {chars_per_sec:.2f} chars/s")

    except Exception as e:
        print(f"❌ Failed to generate Chinese: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 60)
    print("🎉 All tests passed!")
    print("=" * 60)
    print("\nvLLM-Ascend is working on your Ascend 310B4!")
    print("\nNext steps:")
    print("  1. Start vLLM server: vllm serve Qwen/Qwen2.5-0.5B-Instruct")
    print("  2. Run chat app: python app_vllm.py")
    print("=" * 60)

    return True

if __name__ == "__main__":
    success = test_vllm()
    sys.exit(0 if success else 1)

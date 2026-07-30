"""
Run this after installing requirements.txt and paste the output into your
team chat. Compare with your teammate's output before running any pipeline
step — mismatched library versions are a common, silent source of
irreproducible results between two machines.

Usage:
    python scripts/environment_check.py
"""
import platform
import sys


def safe_version(module_name):
    try:
        mod = __import__(module_name)
        return getattr(mod, "__version__", "unknown version")
    except ImportError:
        return "NOT INSTALLED"


def main():
    print("=" * 60)
    print("RetinaVision — Environment Check")
    print("=" * 60)
    print(f"Python version : {platform.python_version()}")
    print(f"Platform       : {platform.platform()}")
    print("-" * 60)

    for pkg in ["numpy", "pandas", "PIL", "imagehash", "torch", "torchvision", "timm", "sklearn", "scipy"]:
        print(f"{pkg:15s}: {safe_version(pkg)}")

    print("-" * 60)
    try:
        import torch
        print(f"CUDA available : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device    : {torch.cuda.get_device_name(0)}")
            print(f"CUDA version   : {torch.version.cuda}")
    except ImportError:
        print("torch not installed — cannot check CUDA.")

    print("=" * 60)
    print("Copy everything above into your team chat and compare with")
    print("your teammate's output before running pipeline scripts.")
    print("=" * 60)


if __name__ == "__main__":
    main()

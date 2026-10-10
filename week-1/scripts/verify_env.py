
import importlib
import sys

LIBS = ["pandas", "sentence_transformers", "chromadb", "spacy"]


def check_gpu() -> None:
    import torch

    assert torch.cuda.is_available(), (
        "CUDA is not available. Check your NVIDIA driver and that you "
        "installed a CUDA build of PyTorch."
    )
    name = torch.cuda.get_device_name(0)
    print(f"[ok] CUDA available | device: {name} | torch {torch.__version__} "
          f"| CUDA {torch.version.cuda}")


def check_imports() -> None:
    for lib in LIBS:
        module = importlib.import_module(lib)
        version = getattr(module, "__version__", "unknown")
        print(f"[ok] import {lib} (version {version})")


def check_basic_use() -> None:
    """Tiny functional checks, so a bad install is caught early."""
    import chromadb
    import spacy

    client = chromadb.EphemeralClient()
    col = client.create_collection("env_check")
    col.add(ids=["1"], documents=["hello world"], embeddings=[[0.1, 0.2, 0.3]])
    assert col.count() == 1
    print("[ok] ChromaDB in-memory collection works")

    nlp = spacy.blank("en")
    assert len(nlp("Hello world")) == 2
    print("[ok] spaCy tokenizer works")


def main() -> int:
    checks = [check_gpu, check_imports, check_basic_use]
    failed = 0
    for check in checks:
        try:
            check()
        except Exception as exc:  # report every failure, not just the first
            failed += 1
            print(f"[FAIL] {check.__name__}: {exc}")
    if failed:
        print(f"\n{failed} check(s) failed.")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

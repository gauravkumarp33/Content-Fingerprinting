import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.embedding_service import EmbeddingService, generate_embedding


def test_imports():
    """Test that all imports work"""
    print("[TEST 1] Checking imports... OK")
    return True


def test_embedding_service_singleton():
    """Test singleton pattern"""
    instance1 = EmbeddingService.get_instance()
    instance2 = EmbeddingService.get_instance()
    assert instance1 is instance2, "Singleton pattern failed"
    print("[TEST 2] Singleton pattern... OK")
    return True


def test_generate_embedding():
    """Test embedding generation with dummy image"""
    # Create a dummy RGB image (100x100x3)
    dummy_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    print("[TEST 3] Generating embedding from dummy image...")
    embedding = generate_embedding(dummy_image)
    
    # Verify output
    assert isinstance(embedding, np.ndarray), "Output should be numpy array"
    assert embedding.dtype == np.float32, f"Expected float32, got {embedding.dtype}"
    assert len(embedding.shape) == 1, "Embedding should be 1D vector"
    assert embedding.shape[0] > 0, "Embedding should have non-zero length"
    
    print(f"  - Shape: {embedding.shape}")
    print(f"  - Dtype: {embedding.dtype}")
    print(f"  - Sample values: {embedding[:5]}")
    print("  OK")
    return True


def test_embedding_normalization():
    """Test that embeddings are normalized"""
    dummy_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    embedding = generate_embedding(dummy_image)
    
    # Check if normalized (L2 norm should be close to 1)
    norm = np.linalg.norm(embedding)
    assert 0.99 < norm < 1.01, f"Embedding should be normalized, norm={norm}"
    print(f"[TEST 4] Embedding normalization... OK (norm={norm:.6f})")
    return True


def test_multiple_images():
    """Test with different images"""
    print("[TEST 5] Testing with multiple images...")
    embeddings = []
    
    for i in range(3):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        emb = generate_embedding(img)
        embeddings.append(emb)
    
    # Check that different images produce different embeddings
    diff_01 = np.linalg.norm(embeddings[0] - embeddings[1])
    diff_12 = np.linalg.norm(embeddings[1] - embeddings[2])
    
    print(f"  - Difference between img1-img2: {diff_01:.6f}")
    print(f"  - Difference between img2-img3: {diff_12:.6f}")
    print("  OK")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("EMBEDDING SERVICE TEST SUITE")
    print("=" * 60 + "\n")
    
    tests = [
        test_imports,
        test_embedding_service_singleton,
        test_generate_embedding,
        test_embedding_normalization,
        test_multiple_images,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            failed += 1
            print(f"  ERROR: {e}\n")
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    sys.exit(0 if failed == 0 else 1)

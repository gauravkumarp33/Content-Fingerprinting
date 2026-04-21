import os
import sys
import tempfile
from pathlib import Path
import time

import cv2
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.media_pipeline import load_media, IMAGE_EXTENSIONS, VIDEO_EXTENSIONS


def test_supported_extensions():
    """Test that extension constants are defined"""
    print("[TEST 1] Checking supported extensions...")
    assert IMAGE_EXTENSIONS, "IMAGE_EXTENSIONS should not be empty"
    assert VIDEO_EXTENSIONS, "VIDEO_EXTENSIONS should not be empty"
    assert ".jpg" in IMAGE_EXTENSIONS, "jpg should be supported"
    assert ".mp4" in VIDEO_EXTENSIONS, "mp4 should be supported"
    print("  ✓ Extensions defined correctly")
    return True


def test_load_image():
    """Test loading a dummy image"""
    print("[TEST 2] Testing image loading...")

    # Create a temporary image file
    tmp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    tmp_file.close()  # Close file handle immediately

    try:
        # Create a simple 100x100 RGB image
        dummy_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(tmp_file.name, dummy_image)

        # Test loading
        frames = load_media(tmp_file.name)
        assert len(frames) == 1, f"Expected 1 frame, got {len(frames)}"
        assert isinstance(frames[0], np.ndarray), "Frame should be numpy array"
        assert frames[0].shape == (100, 100, 3), f"Wrong shape: {frames[0].shape}"
        print(f"  ✓ Image loaded successfully: shape={frames[0].shape}")
        return True
    finally:
        # Clean up with retry for Windows file locking
        for _ in range(5):
            try:
                os.unlink(tmp_file.name)
                break
            except OSError:
                time.sleep(0.1)


def test_load_video():
    """Test loading video frames"""
    print("[TEST 3] Testing video frame extraction...")

    # Create a temporary video file
    tmp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    tmp_file.close()  # Close file handle immediately

    try:
        # Create a simple video with 10 frames
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(tmp_file.name, fourcc, 5.0, (100, 100))  # 5 FPS

        for i in range(10):
            frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            out.write(frame)
        out.release()

        # Test loading
        frames = load_media(tmp_file.name)
        assert len(frames) > 0, "Should extract at least one frame"
        assert all(isinstance(f, np.ndarray) for f in frames), "All frames should be numpy arrays"
        print(f"  ✓ Video processed: {len(frames)} frames extracted")
        return True
    finally:
        # Clean up with retry for Windows file locking
        for _ in range(5):
            try:
                os.unlink(tmp_file.name)
                break
            except OSError:
                time.sleep(0.1)


def test_unsupported_extension():
    """Test error handling for unsupported file types"""
    print("[TEST 4] Testing unsupported file types...")

    tmp_file = tempfile.NamedTemporaryFile(suffix='.txt', delete=False)
    try:
        tmp_file.write(b"Hello World")
        tmp_file.close()

        try:
            load_media(tmp_file.name)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Unsupported media type" in str(e), f"Wrong error message: {e}"
            print("  ✓ Correctly rejected unsupported file type")
            return True
    finally:
        for _ in range(5):
            try:
                os.unlink(tmp_file.name)
                break
            except OSError:
                time.sleep(0.1)


def test_invalid_image():
    """Test error handling for invalid image files"""
    print("[TEST 5] Testing invalid image handling...")

    tmp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    try:
        # Write invalid image data
        tmp_file.write(b"This is not an image")
        tmp_file.close()

        try:
            load_media(tmp_file.name)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Unable to read image" in str(e), f"Wrong error message: {e}"
            print("  ✓ Correctly handled invalid image")
            return True
    finally:
        for _ in range(5):
            try:
                os.unlink(tmp_file.name)
                break
            except OSError:
                time.sleep(0.1)


def test_invalid_video():
    """Test error handling for invalid video files"""
    print("[TEST 6] Testing invalid video handling...")

    tmp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    try:
        # Write invalid video data
        tmp_file.write(b"This is not a video")
        tmp_file.close()

        try:
            load_media(tmp_file.name)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Unable to read video" in str(e), f"Wrong error message: {e}"
            print("  ✓ Correctly handled invalid video")
            return True
    finally:
        for _ in range(5):
            try:
                os.unlink(tmp_file.name)
                break
            except OSError:
                time.sleep(0.1)


def test_video_no_frames():
    """Test handling of videos with no extractable frames"""
    print("[TEST 7] Testing video with no frames...")

    tmp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    tmp_file.close()

    try:
        # Create an empty video file
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(tmp_file.name, fourcc, 1.0, (100, 100))
        # Don't write any frames
        out.release()

        try:
            load_media(tmp_file.name)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            # Accept either "Unable to read video" or "No frames extracted"
            error_msg = str(e)
            assert ("Unable to read video" in error_msg or "No frames extracted" in error_msg), f"Wrong error message: {e}"
            print("  ✓ Correctly handled video with no frames")
            return True
    finally:
        for _ in range(5):
            try:
                os.unlink(tmp_file.name)
                break
            except OSError:
                time.sleep(0.1)


if __name__ == "__main__":
    print("=" * 60)
    print("MEDIA PIPELINE TEST SUITE")
    print("=" * 60 + "\n")

    tests = [
        test_supported_extensions,
        test_load_image,
        test_load_video,
        test_unsupported_extension,
        test_invalid_image,
        test_invalid_video,
        test_video_no_frames,
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
"""Tests for CUDA variant detection logic.

These tests verify that _detect_cuda_variant() correctly maps runtime CUDA
versions to the nearest compatible supported variant. No CUDA hardware needed -
all tests mock torch.version.cuda.
"""

from unittest.mock import patch

import pytest

from fussim import _detect_cuda_variant


class TestDetectCudaVariantExactMatch:
    """Test that exact CUDA version matches work."""

    @pytest.mark.parametrize(
        "cuda_version, expected_variant",
        [
            ("11.8", "cu118"),
            ("12.1", "cu121"),
            ("12.4", "cu124"),
            ("12.6", "cu126"),
            ("12.8", "cu128"),
            ("13.0", "cu130"),
        ],
    )
    def test_exact_match(self, cuda_version, expected_variant):
        with patch("fussim.torch") as mock_torch:
            mock_torch.version.cuda = cuda_version
            assert _detect_cuda_variant() == expected_variant


class TestDetectCudaVariantNearestLower:
    """Test that non-exact CUDA versions fall back to the nearest lower compatible variant.

    CUDA has minor version forward compatibility: a binary compiled against
    CUDA 12.8 runs fine on a 12.9 runtime. So for any CUDA version not in
    the supported list, we should pick the highest supported variant that
    is <= the runtime version.
    """

    @pytest.mark.parametrize(
        "cuda_version, expected_variant",
        [
            # CUDA 12.9 (conda-forge ships this) should use cu128
            ("12.9", "cu128"),
            # CUDA 12.5 should use cu124
            ("12.5", "cu124"),
            # CUDA 12.3 should use cu121
            ("12.3", "cu121"),
            # CUDA 12.2 should use cu121
            ("12.2", "cu121"),
            # CUDA 12.7 should use cu126
            ("12.7", "cu126"),
            # CUDA 13.1 should use cu130
            ("13.1", "cu130"),
            # CUDA 13.2 should use cu130
            ("13.2", "cu130"),
        ],
    )
    def test_nearest_lower_variant(self, cuda_version, expected_variant):
        with patch("fussim.torch") as mock_torch:
            mock_torch.version.cuda = cuda_version
            result = _detect_cuda_variant()
            assert result == expected_variant, (
                f"CUDA {cuda_version} should map to {expected_variant}, got {result}"
            )

    def test_cuda_12_9_does_not_return_none(self):
        """Regression test: CUDA 12.9 must not return None (the original bug)."""
        with patch("fussim.torch") as mock_torch:
            mock_torch.version.cuda = "12.9"
            result = _detect_cuda_variant()
            assert result is not None, (
                "CUDA 12.9 returned None - no compatible variant found. "
                "cu128 should be compatible via CUDA minor version forward compat."
            )


class TestDetectCudaVariantCrossMajor:
    """Test that cross-major-version fallback is NOT allowed.

    CUDA does not guarantee ABI compatibility across major versions.
    A binary compiled against CUDA 13.x should not be loaded on a 14.x runtime,
    and a 11.x binary should not be returned for a 12.x runtime.
    """

    @pytest.mark.parametrize(
        "cuda_version",
        [
            # Future CUDA 14.x - no cu14x variant exists, and cu130 must NOT be returned
            "14.0",
            "14.2",
            # Future CUDA 15.0
            "15.0",
        ],
    )
    def test_future_major_version_returns_none(self, cuda_version):
        """A future major CUDA version with no matching variant should return None."""
        with patch("fussim.torch") as mock_torch:
            mock_torch.version.cuda = cuda_version
            result = _detect_cuda_variant()
            assert result is None, (
                f"CUDA {cuda_version} should return None (no same-major variant), got {result}"
            )

    def test_cuda_12_0_does_not_cross_to_11x(self):
        """CUDA 12.0 is below cu121 but must NOT fall back to cu118 (different major)."""
        with patch("fussim.torch") as mock_torch:
            mock_torch.version.cuda = "12.0"
            result = _detect_cuda_variant()
            assert result is None, (
                f"CUDA 12.0 should return None (no cu12x variant <= 12.0), got {result}"
            )


class TestDetectCudaVariantEdgeCases:
    """Test edge cases for variant detection."""

    def test_no_cuda_returns_none(self):
        with patch("fussim.torch") as mock_torch:
            mock_torch.version.cuda = None
            assert _detect_cuda_variant() is None

    def test_cuda_too_old_returns_none(self):
        """CUDA 10.x has no supported variant, should return None."""
        with patch("fussim.torch") as mock_torch:
            mock_torch.version.cuda = "10.2"
            assert _detect_cuda_variant() is None

    def test_cuda_11_7_returns_none(self):
        """CUDA 11.7 is below cu118, should return None."""
        with patch("fussim.torch") as mock_torch:
            mock_torch.version.cuda = "11.7"
            assert _detect_cuda_variant() is None

    def test_torch_exception_returns_none(self):
        """If torch raises an exception, should gracefully return None."""
        with patch("fussim.torch") as mock_torch:
            type(mock_torch.version).cuda = property(lambda self: (_ for _ in ()).throw(RuntimeError("no cuda")))
            assert _detect_cuda_variant() is None

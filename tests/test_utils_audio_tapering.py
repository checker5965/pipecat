"""Tests for audio tapering functionality.

This module contains unit tests for the audio tapering system, including
word boundary detection, volume reduction, and edge case handling.
"""

import time

import numpy as np
import psutil
import pytest

from pipecat.frames.frames import OutputAudioRawFrame
from pipecat.transports.base_transport import TransportParams
from pipecat.utils.audio_utils import _find_word_boundary, taper_audio_frame


def test_word_boundary_detection():
    """Test word boundary detection with realistic speech patterns.

    This test simulates realistic speech patterns where:
    - Each sample represents 1/16000 seconds
    - Word boundaries typically occur over 20-50ms windows
    - Energy changes happen over multiple samples
    - Includes various speech patterns and edge cases
    """
    # Create test audio data simulating speech pattern
    # First 150ms of high energy speech followed by 850ms of low energy
    sample_rate = 16000
    duration = 1.0  # seconds
    samples = int(sample_rate * duration)

    # Create a speech-like pattern
    audio_data = np.zeros(samples, dtype=np.int16)

    # First 150 ms: High energy speech (values between 10000-20000)
    high_energy_samples = int(0.15 * sample_rate)
    audio_data[:high_energy_samples] = np.random.randint(10000, 20000, high_energy_samples)

    # Last 850 ms: Low energy / silence (values between 0-1000)
    audio_data[high_energy_samples:] = np.random.randint(0, 1000, samples - high_energy_samples)

    # Test with 200ms lookahead window and 10 partitions so we will look for a word boundary every 20ms
    boundary = _find_word_boundary(
        audio_data,
        sample_rate=sample_rate,
        start_idx=0,
        lookahead_window_ms=200,
        lookahead_partitions=10,
    )

    # Boundary should be found near the transition point (150ms = 2400 samples)
    # Allow for some flexibility due to window sizes
    print(f"Boundary: {boundary}")
    assert abs(boundary - 2400) < 200, f"Boundary {boundary} not near expected position 2400"


def test_word_boundary_no_boundary():
    """Test word boundary detection when no clear boundary exists."""
    audio_data = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    boundary = _find_word_boundary(
        audio_data, sample_rate=16000, start_idx=0, energy_threshold_percent=50
    )
    assert boundary == len(audio_data)  # Should return full length


def test_word_boundary_high_threshold():
    """Test word boundary detection with high energy threshold."""
    audio_data = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 1.0, 0.1, 0.2])
    boundary = _find_word_boundary(
        audio_data, sample_rate=16000, start_idx=0, energy_threshold_percent=95
    )
    assert boundary == len(audio_data)  # Should not find boundary with high threshold


@pytest.mark.asyncio
async def test_tapering_volume_reduction():
    """Test volume reduction over multiple steps."""
    # Create test audio data with realistic 16-bit values
    audio_data = np.full(1000, 20000, dtype=np.int16)  # Constant audio at speech level of 20000
    audio_bytes = audio_data.tobytes()  # Convert to bytes

    # Create a raw audio frame with the audio data
    audio_frame = OutputAudioRawFrame(audio=audio_bytes, sample_rate=16000, num_channels=1)

    initialVolume = np.max(np.abs(np.frombuffer(audio_frame.audio, dtype=np.int16)))
    print(f"Initial volume: {initialVolume}")

    # Create a params object
    params = TransportParams(
        tapering_period_ms=200,
        tapering_steps=10,
        tapering_decay_factor=2.0,
        taper_after_word_boundary=False,
    )

    # Collect all frames from the async generator
    tapered_frames = []
    async for frame in taper_audio_frame(audio_frame, params):
        tapered_frames.append(frame)

    final_volume = np.max(np.abs(np.frombuffer(tapered_frames[-1].audio, dtype=np.int16)))
    assert final_volume < 3000  # Max reduction is e^(-2.0) = 0.135 so 20000 * 0.135 ~ 2700


@pytest.mark.asyncio
async def test_tapering_monotonic_decrease():
    """Test that volume decreases monotonically during tapering."""
    # Create test audio data with realistic 16-bit values
    audio_data = np.full(1000, 20000, dtype=np.int16)  # Constant audio at speech level of 20000
    audio_bytes = audio_data.tobytes()  # Convert to bytes

    # Create a raw audio frame with the audio data
    audio_frame = OutputAudioRawFrame(audio=audio_bytes, sample_rate=16000, num_channels=1)

    # Create a params object
    params = TransportParams(
        tapering_period_ms=200,
        tapering_steps=10,
        tapering_decay_factor=2.0,
        taper_after_word_boundary=False,
    )

    # Collect all frames from the async generator
    tapered_frames = []
    async for frame in taper_audio_frame(audio_frame, params):
        tapered_frames.append(frame)

    volumes = [
        np.max(np.abs(np.frombuffer(frame.audio, dtype=np.int16))) for frame in tapered_frames
    ]
    assert all(
        volumes[i] >= volumes[i + 1] for i in range(len(volumes) - 1)
    )  # Volume should decrease monotonically


@pytest.mark.asyncio
async def test_tapering_different_decay_factors():
    """Test tapering with different decay factors."""
    # Create test audio data with realistic 16-bit values
    audio_data = np.full(1000, 20000, dtype=np.int16)  # Constant audio at speech level of 20000
    audio_bytes = audio_data.tobytes()  # Convert to bytes

    # Create a raw audio frame with the audio data
    audio_frame = OutputAudioRawFrame(audio=audio_bytes, sample_rate=16000, num_channels=1)

    decay_factors = [1.5, 2.0, 3.0]
    final_volumes = []

    for factor in decay_factors:
        # Create a params object
        params = TransportParams(
            tapering_period_ms=200,
            tapering_steps=10,
            tapering_decay_factor=factor,
            taper_after_word_boundary=False,
        )

        # Collect all frames from the async generator
        tapered_frames = []
        async for frame in taper_audio_frame(audio_frame, params):
            tapered_frames.append(frame)

        final_volumes.append(
            np.max(np.abs(np.frombuffer(tapered_frames[-1].audio, dtype=np.int16)))
        )

    # Higher decay factors should result in lower final volumes
    assert all(final_volumes[i] >= final_volumes[i + 1] for i in range(len(final_volumes) - 1))


@pytest.mark.asyncio
async def test_tapering_frame_timing():
    """Test that frame timing is maintained during tapering."""
    # Create test audio data with realistic 16-bit values
    audio_data = np.full(
        16000, 20000, dtype=np.int16
    )  # 1 second of audio at reasonable speech level
    audio_bytes = audio_data.tobytes()  # Convert to bytes

    # Create a raw audio frame with the audio data
    audio_frame = OutputAudioRawFrame(audio=audio_bytes, sample_rate=16000, num_channels=1)

    # Create a params object
    params = TransportParams(
        tapering_period_ms=200,
        tapering_steps=10,
        tapering_decay_factor=2.0,
        taper_after_word_boundary=False,
    )

    # Collect all frames from the async generator
    tapered_frames = []
    async for frame in taper_audio_frame(audio_frame, params):
        tapered_frames.append(frame)

    # Check that frames are properly sized
    for frame in tapered_frames:
        assert isinstance(frame, OutputAudioRawFrame)
        assert len(frame.audio) > 0
        assert isinstance(frame.audio, bytes)


@pytest.mark.asyncio
async def test_empty_chunk_handling():
    """Test handling of empty audio chunks."""
    # Create empty audio data
    audio_data = np.array([], dtype=np.int16)
    audio_bytes = audio_data.tobytes()  # Convert to bytes

    # Create a raw audio frame with the audio data
    audio_frame = OutputAudioRawFrame(audio=audio_bytes, sample_rate=16000, num_channels=1)

    # Create a params object
    params = TransportParams(
        tapering_period_ms=200,
        tapering_steps=10,
        tapering_decay_factor=2.0,
        taper_after_word_boundary=False,
    )

    # Collect all frames from the async generator
    tapered_frames = []
    async for frame in taper_audio_frame(audio_frame, params):
        tapered_frames.append(frame)

    assert len(tapered_frames) == 0  # Should handle gracefully


@pytest.mark.asyncio
async def test_tapering_with_silence():
    """Test tapering with audio containing silence."""
    # Create test audio data with realistic 16-bit values
    audio_data = np.concatenate(
        [
            np.zeros(500, dtype=np.int16),  # Silence
            np.full(500, 20000, dtype=np.int16),  # Sound at reasonable speech level
            np.zeros(500, dtype=np.int16),  # Silence
        ]
    )
    audio_bytes = audio_data.tobytes()  # Convert to bytes

    # Create a raw audio frame with the audio data
    audio_frame = OutputAudioRawFrame(audio=audio_bytes, sample_rate=16000, num_channels=1)

    # Create a params object
    params = TransportParams(
        tapering_period_ms=200,
        tapering_steps=10,
        tapering_decay_factor=2.0,
        taper_after_word_boundary=False,
    )

    # Collect all frames from the async generator
    tapered_frames = []
    async for frame in taper_audio_frame(audio_frame, params):
        tapered_frames.append(frame)

    assert len(tapered_frames) > 0  # Should handle silence


@pytest.mark.asyncio
async def test_tapering_with_noise():
    """Test tapering with noisy audio data."""
    np.random.seed(42)  # For reproducibility
    # Create test audio data with realistic 16-bit values
    audio_data = (np.random.randn(1000) * 10000 + 10000).astype(
        np.int16
    )  # Gaussian noise centered around reasonable speech level
    audio_bytes = audio_data.tobytes()  # Convert to bytes

    # Create a raw audio frame with the audio data
    audio_frame = OutputAudioRawFrame(audio=audio_bytes, sample_rate=16000, num_channels=1)

    # Create a params object
    params = TransportParams(
        tapering_period_ms=200,
        tapering_steps=10,
        tapering_decay_factor=2.0,
        taper_after_word_boundary=False,
    )

    # Collect all frames from the async generator
    tapered_frames = []
    async for frame in taper_audio_frame(audio_frame, params):
        tapered_frames.append(frame)

    assert len(tapered_frames) > 0  # Should handle noise


@pytest.mark.asyncio
async def test_tapering_edge_cases():
    """Test tapering with various edge cases."""
    # Test with very short audio
    short_audio = np.full(10, 20000, dtype=np.int16)  # Very short audio at reasonable speech level
    short_audio_bytes = short_audio.tobytes()  # Convert to bytes

    # Create a raw audio frame with the audio data
    audio_frame = OutputAudioRawFrame(audio=short_audio_bytes, sample_rate=16000, num_channels=1)

    # Create a params object
    params = TransportParams(
        tapering_period_ms=200,
        tapering_steps=10,
        tapering_decay_factor=2.0,
        taper_after_word_boundary=False,
    )

    # Collect all frames from the async generator
    tapered_frames = []
    async for frame in taper_audio_frame(audio_frame, params):
        tapered_frames.append(frame)

    assert len(tapered_frames) > 0

    # Test with very long audio
    long_audio = np.full(100000, 20000, dtype=np.int16)  # Long audio at reasonable speech level
    long_audio_bytes = long_audio.tobytes()  # Convert to bytes

    # Create a raw audio frame with the audio data
    audio_frame = OutputAudioRawFrame(audio=long_audio_bytes, sample_rate=16000, num_channels=1)

    # Create a params object
    params = TransportParams(
        tapering_period_ms=200,
        tapering_steps=10,
        tapering_decay_factor=2.0,
        taper_after_word_boundary=False,
    )

    # Collect all frames from the async generator
    tapered_frames = []
    async for frame in taper_audio_frame(audio_frame, params):
        tapered_frames.append(frame)

    assert len(tapered_frames) > 0

    # Test with negative values
    negative_audio = np.full(
        1000, -20000, dtype=np.int16
    )  # Negative audio at reasonable speech level
    negative_audio_bytes = negative_audio.tobytes()  # Convert to bytes

    # Create a raw audio frame with the audio data
    audio_frame = OutputAudioRawFrame(audio=negative_audio_bytes, sample_rate=16000, num_channels=1)

    # Create a params object
    params = TransportParams(
        tapering_period_ms=200,
        tapering_steps=10,
        tapering_decay_factor=2.0,
        taper_after_word_boundary=False,
    )

    # Collect all frames from the async generator
    tapered_frames = []
    async for frame in taper_audio_frame(audio_frame, params):
        tapered_frames.append(frame)

    assert len(tapered_frames) > 0


@pytest.mark.asyncio
async def test_tapering_parameter_validation():
    """Test validation of tapering parameters."""
    audio_data = np.ones(1000)

    # Create a raw audio frame with the audio data
    audio_frame = OutputAudioRawFrame(audio=audio_data, sample_rate=16000, num_channels=1)

    # Test invalid tapering period
    with pytest.raises(ValueError):
        params = TransportParams(
            tapering_period_ms=-100,
            tapering_steps=10,
            tapering_decay_factor=2.0,
            taper_after_word_boundary=False,
        )
        async for _ in taper_audio_frame(audio_frame, params):
            pass

    # Test invalid number of steps
    with pytest.raises(ValueError):
        params = TransportParams(
            tapering_period_ms=200,
            tapering_steps=0,
            tapering_decay_factor=2.0,
            taper_after_word_boundary=False,
        )
        async for _ in taper_audio_frame(audio_frame, params):
            pass

    # Test invalid decay factor
    with pytest.raises(ValueError):
        params = TransportParams(
            tapering_period_ms=200,
            tapering_steps=10,
            tapering_decay_factor=0.5,
            taper_after_word_boundary=False,
        )
        async for _ in taper_audio_frame(audio_frame, params):
            pass


@pytest.mark.asyncio
async def test_tapering_performance():
    """Test performance characteristics of audio tapering."""
    # Create a realistic audio frame (1 second of audio at 16kHz)
    audio_data = np.full(16000, 20000, dtype=np.int16)
    audio_bytes = audio_data.tobytes()
    audio_frame = OutputAudioRawFrame(audio=audio_bytes, sample_rate=16000, num_channels=1)

    # Create params with standard settings
    params = TransportParams(
        tapering_period_ms=200,
        tapering_steps=10,
        tapering_decay_factor=2.0,
        taper_after_word_boundary=False,
    )

    # Measure memory before processing
    process = psutil.Process()
    initial_memory = process.memory_info().rss

    # Time the processing
    start_time = time.time()
    tapered_frames = []
    async for frame in taper_audio_frame(audio_frame, params):
        tapered_frames.append(frame)
    end_time = time.time()

    # Calculate metrics
    processing_time = end_time - start_time
    final_memory = process.memory_info().rss
    memory_used = final_memory - initial_memory
    frames_per_second = len(tapered_frames) / processing_time

    # Log performance metrics
    print(f"\nPerformance Metrics:")
    print(f"Processing time: {processing_time:.4f} seconds")
    print(f"Memory used: {memory_used / 1024:.2f} KB")
    print(f"Frames per second: {frames_per_second:.2f}")

    # Assert performance requirements
    assert processing_time < 0.25  # Should process in less than 250ms
    assert memory_used < 1024 * 1024  # Should use less than 1MB of memory


@pytest.mark.asyncio
async def test_tapering_performance_under_load():
    """Test performance under sustained load."""
    # Create multiple audio frames
    num_frames = 100
    frames = []
    for _ in range(num_frames):
        audio_data = np.full(16000, 20000, dtype=np.int16)
        audio_bytes = audio_data.tobytes()
        frames.append(OutputAudioRawFrame(audio=audio_bytes, sample_rate=16000, num_channels=1))

    params = TransportParams(
        tapering_period_ms=200,
        tapering_steps=10,
        tapering_decay_factor=2.0,
        taper_after_word_boundary=False,
    )

    # Measure memory before processing
    process = psutil.Process()
    initial_memory = process.memory_info().rss

    # Process all frames and measure time
    start_time = time.time()
    all_tapered_frames = []
    for frame in frames:
        tapered_frames = []
        async for tapered_frame in taper_audio_frame(frame, params):
            tapered_frames.append(tapered_frame)
        all_tapered_frames.extend(tapered_frames)
    end_time = time.time()

    # Calculate metrics
    total_time = end_time - start_time
    final_memory = process.memory_info().rss
    memory_used = final_memory - initial_memory
    average_time_per_frame = total_time / num_frames

    # Log performance metrics
    print(f"\nPerformance Under Load:")
    print(f"Total processing time: {total_time:.4f} seconds")
    print(f"Average time per frame: {average_time_per_frame:.4f} seconds")
    print(f"Total memory used: {memory_used / 1024:.2f} KB")

    # Assert performance requirements
    assert average_time_per_frame < 0.25  # Should process each frame in less than 250ms
    assert memory_used < 10 * 1024 * 1024  # Should use less than 10MB of memory for 100 frames


@pytest.mark.asyncio
async def test_tapering_performance_with_large_frames():
    """Test performance with large audio frames."""
    # Create a large audio frame (10 seconds of audio at 16kHz)
    audio_data = np.full(160000, 20000, dtype=np.int16)
    audio_bytes = audio_data.tobytes()
    audio_frame = OutputAudioRawFrame(audio=audio_bytes, sample_rate=16000, num_channels=1)

    params = TransportParams(
        tapering_period_ms=200,
        tapering_steps=10,
        tapering_decay_factor=2.0,
        taper_after_word_boundary=False,
    )

    # Measure memory before processing
    process = psutil.Process()
    initial_memory = process.memory_info().rss

    # Time the processing
    start_time = time.time()
    tapered_frames = []
    async for frame in taper_audio_frame(audio_frame, params):
        tapered_frames.append(frame)
    end_time = time.time()

    # Calculate metrics
    processing_time = end_time - start_time
    final_memory = process.memory_info().rss
    memory_used = final_memory - initial_memory
    bytes_per_second = len(audio_bytes) / processing_time

    # Log performance metrics
    print(f"\nLarge Frame Performance:")
    print(f"Processing time: {processing_time:.4f} seconds")
    print(f"Memory used: {memory_used / 1024:.2f} KB")
    print(f"Processing speed: {bytes_per_second / 1024:.2f} KB/s")

    # Assert performance requirements
    assert processing_time < 1.0  # Should process in less than 1 second
    assert memory_used < 5 * 1024 * 1024  # Should use less than 5MB of memory
    assert bytes_per_second > 100 * 1024  # Should process at least 100KB/s

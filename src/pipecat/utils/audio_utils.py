"""Audio utility functions for Pipecat.

This module provides audio processing utilities, including:
- Word boundary detection for natural speech interruption
- Audio tapering for smooth volume reduction
- Volume scaling calculations
"""

import asyncio
from typing import AsyncGenerator

import numpy as np
from loguru import logger

from pipecat.frames.frames import OutputAudioRawFrame

MAX_LOOKAHEAD_WINDOW_MS = 500
MAX_LOOKAHEAD_PARTITIONS = 10
MIN_ENERGY_THRESHOLD_PERCENT = 10
MAX_ENERGY_THRESHOLD_PERCENT = 50


def _find_word_boundary(
    input_audio: np.ndarray,
    sample_rate: int,
    start_idx: int,
    lookahead_window_ms: int = 100,
    lookahead_partitions: int = 5,
    energy_threshold_percent: float = 30,
) -> int:
    """Find the next word boundary in the audio data.

    This function implements a configurable word boundary detection algorithm that:
    1. Analyzes audio energy in configurable time windows
    2. Identifies points of significant energy drop based on a configurable threshold
    3. Returns the sample index of the first detected boundary

    Args:
        input_audio (np.ndarray): Input audio data as a numpy array of int16 samples
        sample_rate (int): Audio sample rate in Hz
        start_idx (int): Starting index in the audio array to begin analysis
        lookahead_window_ms (int): Look-ahead window in milliseconds for boundary detection.
                                  Default: 100ms. Maximum: 500ms.
        lookahead_partitions (int): Number of partitions to divide the look-ahead window into.
                                   Default: 5. Range: 1-10.
        energy_threshold_percent (float): Energy threshold as percentage of average energy.
                                         Default: 30%. Range: 10-50%.

    Returns:
        int: Sample index of the first detected word boundary, or the end of the
             look-ahead window if no boundary is found

    Notes:
        - The look-ahead window is clamped between 0ms and MAX_LOOKAHEAD_WINDOW_MS (500ms)
        - The number of partitions is clamped between 1 and MAX_LOOKAHEAD_PARTITIONS (10)
        - The energy threshold is clamped between MIN_ENERGY_THRESHOLD_PERCENT (10%) and
          MAX_ENERGY_THRESHOLD_PERCENT (50%)
        - Window size is calculated as: lookahead_window_ms / lookahead_partitions
        - Returns the end of the look-ahead window if no clear boundary is found

    Example:
        >>> audio = np.array([1000, 2000, 3000, 100, 200, 300], dtype=np.int16)
        >>> # Use default parameters
        >>> boundary = _find_word_boundary(audio, 16000, 0)
        >>> # Use custom parameters
        >>> boundary = _find_word_boundary(audio, 16000, 0,
        ...                               lookahead_window_ms=200,
        ...                               lookahead_partitions=8,
        ...                               energy_threshold_percent=25)
        >>> print(f"Word boundary at sample {boundary}")
    """
    # Clamp the lookahead window between 0 and the maximum allowed value and convert to seconds
    lookahead_window_seconds = max(
        0, min(lookahead_window_ms / 1000, MAX_LOOKAHEAD_WINDOW_MS / 1000)
    )
    # Clamp the number of partitions between 1 and the maximum allowed value
    lookahead_partitions = max(1, min(lookahead_partitions, MAX_LOOKAHEAD_PARTITIONS))
    # Clamp the energy threshold between 1 and the maximum allowed value
    energy_threshold_percent = max(
        MIN_ENERGY_THRESHOLD_PERCENT, min(energy_threshold_percent, MAX_ENERGY_THRESHOLD_PERCENT)
    )
    # Look ahead for a word boundary
    look_ahead_samples = int(sample_rate * lookahead_window_seconds)
    end_idx = min(start_idx + look_ahead_samples, len(input_audio))

    # Calculate energy in small windows
    window_size = int(sample_rate * lookahead_window_seconds / lookahead_partitions)
    energy = []
    for i in range(start_idx, end_idx, window_size):
        window = input_audio[i : min(i + window_size, end_idx)]
        energy.append(np.mean(np.abs(window)))  # Calculate the mean absolute value of the window

    # Find the first point where energy drops significantly
    threshold = np.mean(energy) * energy_threshold_percent / 100
    for i, e in enumerate(energy):
        if e < threshold:
            return start_idx + (
                i * window_size
            )  # Return the index of the first point where energy drops significantly

    return end_idx


async def taper_audio_frame(
    frame: OutputAudioRawFrame, params
) -> AsyncGenerator[OutputAudioRawFrame, None]:
    """Taper off an audio frame exponentially over the specified period.

    This function implements a natural-sounding audio tapering effect that:
    1. Optionally attempts to complete the current word before starting the taper
    2. Applies an exponential decay to the remaining audio
    3. Handles empty or invalid audio frames gracefully

    Args:
        frame (OutputAudioRawFrame): The audio frame to taper. Contains:
            - audio: Raw audio data bytes
            - sample_rate: Audio sample rate in Hz
            - num_channels: Number of audio channels
        params: Object containing tapering parameters:
            - tapering_period_ms: Total duration of the taper in milliseconds
            - tapering_steps: Number of steps in the taper
            - tapering_decay_factor: Controls the steepness of the volume drop
            - taper_after_word_boundary (bool): If True, attempts to find and complete the current
                                         word before starting the taper. If False, starts
                                         tapering immediately. Default: True.

    Yields:
        OutputAudioRawFrame: A series of tapered audio frames with:
            - Gradually decreasing volume
            - Same sample rate and channels as input
            - Audio data scaled by exponential decay

    Raises:
        None: The function handles errors gracefully and logs warnings

    Example:
        >>> async for tapered_frame in taper_audio_frame(audio_frame, params):
        ...     await process_audio(tapered_frame)


    Notes:
        - When params.taper_after_word_boundary is True:
            * First attempts to find a word boundary using _find_word_boundary
            * Outputs audio up to the boundary at full volume
            * Applies tapering to the remaining audio
        - When params.taper_after_word_boundary is False:
            * Starts tapering immediately from the beginning of the frame
        - Volume scaling follows the formula: exp(-decay_factor * (step / (steps-1)))
        - Empty or invalid audio frames are skipped with appropriate warnings
        - The tapering effect is designed to sound natural when speech is interrupted
    """
    audio_data = frame.audio
    sample_rate = frame.sample_rate
    num_channels = frame.num_channels

    # Safety check for empty audio
    if not audio_data or len(audio_data) == 0:
        logger.warning("Empty audio frame received for tapering")
        return

    audio_np = np.frombuffer(audio_data, dtype=np.int16)

    # Safety check for empty numpy array - if audio_data is not empty, this should never happen,
    # but it's good to have the check anyway
    if len(audio_np) == 0:
        logger.warning("Empty audio numpy array after conversion")
        return

    # Log initial audio stats
    initial_volume = np.mean(np.abs(audio_np))
    if np.isnan(initial_volume):
        logger.warning("Initial audio volume is NaN, skipping tapering")
        return

    logger.debug(f"Initial audio volume: {initial_volume}")
    logger.debug(
        f"Tapering params: period={params.tapering_period_ms}ms, steps={params.tapering_steps}, decay={params.tapering_decay_factor}"
    )

    # Find the next word boundary
    word_boundary = None
    if params.taper_after_word_boundary:
        word_boundary = _find_word_boundary(audio_np, sample_rate, 0)

    # First, output the audio up to the word boundary at full volume if it exists
    if word_boundary is not None and word_boundary > 0:
        yield OutputAudioRawFrame(
            audio=audio_np[:word_boundary].tobytes(),
            sample_rate=sample_rate,
            num_channels=num_channels,
        )

    # Then taper the remaining audio
    if word_boundary is not None and word_boundary > 0:
        remaining_audio = audio_np[word_boundary:]
    else:
        remaining_audio = audio_np

    if len(remaining_audio) == 0:
        logger.debug("No remaining audio to taper")
        return

    total_steps = params.tapering_steps
    samples_per_step = len(remaining_audio) // total_steps

    for step in range(total_steps):
        start = step * samples_per_step
        end = (step + 1) * samples_per_step if step < total_steps - 1 else len(remaining_audio)
        chunk = remaining_audio[start:end].copy()

        # Exponential decay
        volume_scale = np.exp(-params.tapering_decay_factor * (step / (total_steps - 1)))
        chunk = (chunk * volume_scale).astype(np.int16)

        # Log volume changes
        current_volume = np.mean(np.abs(chunk))
        if not np.isnan(current_volume):
            logger.debug(
                f"Step {step}/{total_steps}: volume_scale={volume_scale:.3f}, current_volume={current_volume:.3f}"
            )

        # Create a new audio frame with the tapered chunk
        yield OutputAudioRawFrame(
            audio=chunk.tobytes(), sample_rate=sample_rate, num_channels=num_channels
        )

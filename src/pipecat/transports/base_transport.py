#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Base transport module for Pipecat.

This module provides the base classes and interfaces for implementing different
transport mechanisms in Pipecat. It defines the core transport parameters and
abstract base class that all specific transport implementations must inherit from.
"""

from abc import abstractmethod
from typing import Optional

from pydantic import BaseModel, ConfigDict

from pipecat.audio.filters.base_audio_filter import BaseAudioFilter
from pipecat.audio.mixers.base_audio_mixer import BaseAudioMixer
from pipecat.audio.vad.vad_analyzer import VADAnalyzer
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.utils.base_object import BaseObject


class TransportParams(BaseModel):
    """Base class for transport-specific parameters.

    This class defines the common parameters that all transport implementations
    may need, such as audio and camera settings. Specific transport implementations
    should extend this class to add their own parameters.

    Attributes:
        model_config: Pydantic configuration allowing arbitrary types
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    camera_in_enabled: bool = False
    camera_out_enabled: bool = False
    camera_out_is_live: bool = False
    camera_out_width: int = 1024
    camera_out_height: int = 768
    camera_out_bitrate: int = 800000
    camera_out_framerate: int = 30
    camera_out_color_format: str = "RGB"
    audio_out_enabled: bool = False
    audio_out_sample_rate: Optional[int] = None
    audio_out_channels: int = 1
    audio_out_bitrate: int = 96000
    audio_out_10ms_chunks: int = 4
    audio_out_mixer: Optional[BaseAudioMixer] = None
    audio_in_enabled: bool = False
    audio_in_sample_rate: Optional[int] = None
    audio_in_channels: int = 1
    audio_in_filter: Optional[BaseAudioFilter] = None
    audio_in_stream_on_start: bool = True
    vad_enabled: bool = False
    vad_audio_passthrough: bool = False
    vad_analyzer: Optional[VADAnalyzer] = None
    tapering_period_ms: int = 0  # By default, tapering is disabled - the longer the period, the longer the bot will speak before halting
    tapering_steps: int = (
        10  # By default, tapering is done in 10 steps - decrease this to make the taper faster
    )
    tapering_decay_factor: float = 1.2  # By default, the decay factor is 1.2 - increase this to make the taper more aggressive. The final volume becomes e^(-tapering_decay_factor)
    taper_after_word_boundary: bool = True  # By default, tapering is done after the word boundary - set to False to start tapering immediately


class BaseTransport(BaseObject):
    """Abstract base class for all transport implementations.

    This class defines the interface that all transport implementations must
    follow. It provides abstract methods for input and output processing that
    must be implemented by concrete transport classes.

    Attributes:
        params: Transport-specific parameters
    """

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        input_name: Optional[str] = None,
        output_name: Optional[str] = None,
    ):
        """Initialize the base transport.

        Args:
            name: Optional name for the transport.
            input_name: Optional name for the input frame processor.
            output_name: Optional name for the output frame processor.
        """
        super().__init__(name=name)
        self._input_name = input_name
        self._output_name = output_name

    @abstractmethod
    def input(self) -> FrameProcessor:
        """Get the input frame processor for this transport.

        Returns:
            FrameProcessor: A processor that handles incoming frames
        """
        pass

    @abstractmethod
    def output(self) -> FrameProcessor:
        """Get the output frame processor for this transport.

        Returns:
            FrameProcessor: A processor that handles outgoing frames
        """
        pass

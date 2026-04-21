"""Text-to-Speech via Fish Audio API — returns OGG/Opus bytes for Telegram sendVoice."""

import asyncio
import re
from typing import Optional

import structlog

from src.config.settings import Settings

logger = structlog.get_logger(__name__)


class TTSHandler:
    """Convert text to speech using Fish Audio cloud API.

    Produces OGG/Opus audio suitable for Telegram's ``sendVoice`` method.
    If ``ffmpeg`` is available the SDK output (MP3) is transcoded to OGG/Opus;
    otherwise the raw MP3 bytes are returned (Telegram still plays it, but it
    may appear as a file rather than an inline voice bubble).
    """

    # Subprocess timeout for ffmpeg conversion.
    FFMPEG_TIMEOUT: int = 30

    # Maximum text length we'll send to TTS in one shot.
    MAX_TEXT_LENGTH: int = 4000

    def __init__(self, config: Settings) -> None:
        self.config = config
        self._client: Optional[object] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def synthesize(self, text: str) -> Optional[bytes]:
        """Convert *text* to OGG/Opus audio bytes.

        Returns ``None`` when TTS is disabled or the text is empty.
        """
        if not self.config.tts_enabled:
            return None

        text = self._prepare_text(text)
        if not text:
            return None

        try:
            raw_audio = await self._call_fish_audio(text)
            ogg_audio = await self._convert_to_ogg_opus(raw_audio)
            return ogg_audio
        except Exception:
            logger.exception("TTS synthesis failed")
            return None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _prepare_text(self, text: str) -> str:
        """Strip HTML tags and truncate overly long text."""
        # Remove HTML tags (bot responses use HTML formatting)
        clean = re.sub(r"<[^>]+>", "", text)
        # Collapse whitespace
        clean = re.sub(r"\s+", " ", clean).strip()
        # Remove code blocks and technical artifacts
        clean = re.sub(r"```[\s\S]*?```", "", clean)
        clean = re.sub(r"`[^`]+`", "", clean)
        # Truncate
        if len(clean) > self.MAX_TEXT_LENGTH:
            clean = clean[: self.MAX_TEXT_LENGTH] + "…"
        return clean.strip()

    async def _call_fish_audio(self, text: str) -> bytes:
        """Call Fish Audio TTS API and return raw audio bytes (MP3)."""
        client = self._get_client()

        # Fish Audio SDK's convert() is synchronous — run in a thread.
        loop = asyncio.get_running_loop()

        def _do_tts() -> bytes:
            from fish_audio_sdk import TTSRequest

            result = client.tts(
                TTSRequest(
                    text=text,
                    reference_id=self.config.tts_fish_model_id,
                    format="mp3",
                )
            )
            # result is a generator of bytes chunks — collect them all
            return b"".join(result)

        audio_bytes = await loop.run_in_executor(None, _do_tts)

        logger.info(
            "Fish Audio TTS completed",
            text_length=len(text),
            audio_bytes=len(audio_bytes),
        )
        return audio_bytes

    async def _convert_to_ogg_opus(self, mp3_bytes: bytes) -> bytes:
        """Transcode MP3 → OGG/Opus using ffmpeg.

        Falls back to returning the original MP3 bytes if ffmpeg is
        unavailable.
        """
        try:
            process = await asyncio.create_subprocess_exec(
                "ffmpeg",
                "-i", "pipe:0",          # read from stdin
                "-c:a", "libopus",        # Opus codec
                "-b:a", "48k",            # bitrate (voice)
                "-vbr", "on",
                "-application", "voip",   # optimised for speech
                "-f", "ogg",              # OGG container
                "pipe:1",                 # write to stdout
                "-y",
                "-loglevel", "error",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=mp3_bytes),
                timeout=self.FFMPEG_TIMEOUT,
            )

            if process.returncode != 0:
                logger.warning(
                    "ffmpeg OGG/Opus conversion failed, using MP3 fallback",
                    stderr=stderr.decode()[:200],
                )
                return mp3_bytes

            logger.info(
                "Converted MP3 → OGG/Opus",
                mp3_size=len(mp3_bytes),
                ogg_size=len(stdout),
            )
            return stdout

        except FileNotFoundError:
            logger.warning(
                "ffmpeg not found — returning MP3 (voice may display as file)"
            )
            return mp3_bytes

        except asyncio.TimeoutError:
            logger.warning("ffmpeg conversion timed out — returning MP3")
            return mp3_bytes

    def _get_client(self) -> object:
        """Lazy-initialise the Fish Audio SDK client."""
        if self._client is not None:
            return self._client

        try:
            from fish_audio_sdk import Session
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "fish-audio-sdk is required for TTS. "
                "Install it with: pip install fish-audio-sdk"
            ) from exc

        api_key = self.config.tts_fish_api_key_str
        if not api_key:
            raise RuntimeError(
                "Fish Audio API key is not configured (TTS_FISH_API_KEY)."
            )

        self._client = Session(api_key)
        return self._client

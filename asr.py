import io
import wave
import ctypes
from pyogg.opus import (
    opus_decoder_create, opus_decode, opus_decoder_destroy,
    OPUS_OK, c_int, c_int16, c_ubyte,
)

SAMPLE_RATE = 16000
CHANNELS = 1
FRAME_SAMPLES = 960  # 60ms at 16kHz


def decode_opus_frames(frames: list[bytes]) -> bytes:
    """Decode raw Opus frames to raw PCM16LE bytes."""
    err = c_int(0)
    dec = opus_decoder_create(SAMPLE_RATE, CHANNELS, ctypes.byref(err))
    if err.value != OPUS_OK:
        raise RuntimeError(f"opus_decoder_create failed: {err.value}")

    out_buf = (c_int16 * FRAME_SAMPLES)()
    all_pcm = bytearray()

    for frame in frames:
        in_data = (c_ubyte * len(frame))(*frame)
        n = opus_decode(dec, in_data, len(frame), out_buf, FRAME_SAMPLES, 0)
        if n > 0:
            chunk = bytearray(n * 2)
            ctypes.memmove(chunk, out_buf, n * 2)
            all_pcm.extend(chunk)

    opus_decoder_destroy(dec)
    return bytes(all_pcm)


def pcm_to_wav(pcm: bytes) -> bytes:
    """Wrap raw PCM16LE in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm)
    buf.seek(0)
    return buf.read()


async def transcribe(frames: list[bytes], client) -> str:
    """Decode Opus frames and transcribe with OpenAI Whisper.

    Returns empty string if frames is empty or transcription fails.
    """
    if not frames:
        return ""
    try:
        pcm = decode_opus_frames(frames)
        wav = pcm_to_wav(pcm)
        resp = await client.audio.transcriptions.create(
            model="whisper-1",
            file=("audio.wav", io.BytesIO(wav), "audio/wav"),
            language="zh",
        )
        return resp.text.strip()
    except Exception as e:
        from logger import log
        log(f"ASR error: {e}")
        return ""
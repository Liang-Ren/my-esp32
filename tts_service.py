import ctypes
import struct
import miniaudio
import edge_tts
from pyogg.opus import (
    opus_encoder_create, opus_encode, opus_encoder_destroy,
    OPUS_APPLICATION_VOIP, OPUS_OK, c_int, c_int16, c_ubyte,
)

SAMPLE_RATE = 16000
CHANNELS = 1
FRAME_DURATION_MS = 60
FRAME_SAMPLES = SAMPLE_RATE * FRAME_DURATION_MS // 1000  # 960
VOICE = "zh-CN-XiaoxiaoNeural"


def _pcm_to_opus(pcm: bytes) -> list[bytes]:
    err = c_int(0)
    enc = opus_encoder_create(SAMPLE_RATE, CHANNELS, OPUS_APPLICATION_VOIP, ctypes.byref(err))
    if err.value != OPUS_OK:
        raise RuntimeError(f"opus_encoder_create failed: {err.value}")

    frame_bytes = FRAME_SAMPLES * 2
    remainder = len(pcm) % frame_bytes
    if remainder:
        pcm += b"\x00" * (frame_bytes - remainder)

    out_buf = (c_ubyte * 4000)()
    frames = []
    for i in range(0, len(pcm), frame_bytes):
        chunk = pcm[i : i + frame_bytes]
        pcm_arr = (c_int16 * FRAME_SAMPLES)(*struct.unpack_from(f"<{FRAME_SAMPLES}h", chunk))
        n = opus_encode(enc, pcm_arr, FRAME_SAMPLES, out_buf, 4000)
        if n > 0:
            frames.append(bytes(out_buf[:n]))

    opus_encoder_destroy(enc)
    return frames


async def generate(text: str) -> list[bytes]:
    """Convert text to list of raw Opus frames via edge-tts."""
    c = edge_tts.Communicate(text, voice=VOICE)
    mp3 = b""
    async for chunk in c.stream():
        if chunk["type"] == "audio":
            mp3 += chunk["data"]

    decoded = miniaudio.decode(
        mp3,
        output_format=miniaudio.SampleFormat.SIGNED16,
        nchannels=CHANNELS,
        sample_rate=SAMPLE_RATE,
    )
    return _pcm_to_opus(bytes(decoded.samples))


def make_frame(opus_data: bytes, version: int) -> bytes:
    if version == 2:
        return struct.pack(">HHHII", 2, 0, 0, 0, len(opus_data)) + opus_data
    elif version == 3:
        return bytes([0, 0]) + struct.pack(">H", len(opus_data)) + opus_data
    return opus_data  # v1: raw
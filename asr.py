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

_whisper_model = None


def _get_model():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        from logger import log
        log("Loading Whisper model (first time may download ~500MB)...")
        _whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
        log("Whisper model ready")
    return _whisper_model


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
        in_ptr = ctypes.cast(in_data, ctypes.POINTER(c_ubyte))
        n = opus_decode(dec, in_ptr, ctypes.c_long(len(frame)), out_buf, ctypes.c_long(FRAME_SAMPLES), ctypes.c_long(0))
        if n > 0:
            all_pcm += bytes(out_buf)[:n * 2]

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


async def transcribe(frames: list[bytes], client=None) -> str:
    """Decode Opus frames and transcribe with local Whisper.

    Returns empty string if frames is empty or transcription fails.
    """
    if not frames:
        return ""
    try:
        pcm = decode_opus_frames(frames)
        if not pcm:
            return ""
        wav = pcm_to_wav(pcm)
        model = _get_model()
        segments, _ = model.transcribe(
            io.BytesIO(wav),
            language="zh",
            beam_size=5,
            vad_filter=True,
        )
        text = "".join(seg.text for seg in segments).strip()
        return text
    except Exception as e:
        import traceback
        from logger import log
        log(f"ASR error: {e}\n{traceback.format_exc()}")
        return ""
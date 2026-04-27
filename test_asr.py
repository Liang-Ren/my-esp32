#!/usr/bin/env python3
import sys, ctypes
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r'c:\Users\liang\Copilot\.venv\xiaozhi')

from pyogg.opus import (
    opus_decoder_create, opus_decode, opus_decoder_destroy,
    OPUS_OK, c_int, c_int16, c_ubyte,
)
import pyogg.opus as _op

SAMPLE_RATE = 16000
CHANNELS    = 1
FRAME_SAMPLES = 960

# create decoder
err = c_int(0)
dec = opus_decoder_create(SAMPLE_RATE, CHANNELS, ctypes.byref(err))
print(f"dec = {dec}, type = {type(dec)}, err = {err.value}, OPUS_OK = {OPUS_OK}")

# check what types opus_decode's underlying lib function expects
lib_decode = _op.libopus.opus_decode
print(f"libopus.opus_decode.argtypes = {lib_decode.argtypes}")
print(f"libopus.opus_decode.restype  = {lib_decode.restype}")

# make a tiny fake opus frame (silence packet)
fake_frame = bytes([0xf8, 0xff, 0xfe])  # minimal valid opus frame
out_buf = (c_int16 * FRAME_SAMPLES)()

in_data = (c_ubyte * len(fake_frame))(*fake_frame)
in_ptr  = ctypes.cast(in_data, ctypes.POINTER(c_ubyte))

print(f"\nin_ptr type: {type(in_ptr)}")
print(f"out_buf type: {type(out_buf)}")
print(f"dec type: {type(dec)}")

# try decode
try:
    n = opus_decode(dec, in_ptr, ctypes.c_int32(len(fake_frame)), out_buf, ctypes.c_int(FRAME_SAMPLES), ctypes.c_int(0))
    print(f"opus_decode returned: {n}")
except Exception as e:
    print(f"ERROR: {e}")

opus_decoder_destroy(dec)
print("done")
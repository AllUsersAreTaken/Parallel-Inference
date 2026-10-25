"""Build a LoRA adapter that carries a token_embd weight (a non-row-segmentable
tensor), used to test the per-row readiness check.

Per-row (segmented) LoRA decode cannot apply token_embd, output or MoE expert
weights: those tensors bypass build_lora_mm. The generated adapter is
structurally valid (the loader accepts it) but its token_embd pair is zero, so
it must never be used for a real run.

Usage:
  python harness/make_bad_segment_lora.py <base_model.gguf> <src_adapter.gguf> <dst.gguf>

The metadata block is copied from src_adapter (correct general.architecture and
adapter.lora.alpha); the two token_embd tensors are written from scratch with
the shapes the loader validates against the base model.
"""
import struct
import sys

GGML_TYPE_F16 = 1
SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
RANK = 16


def read_string(buf, off):
    (n,) = struct.unpack_from("<Q", buf, off)
    off += 8
    return buf[off:off + n].decode("utf-8"), off + n


def skip_value(buf, off, t):
    if t == 8:
        return read_string(buf, off)[1]
    if t == 9:
        (et,) = struct.unpack_from("<I", buf, off)
        (n,) = struct.unpack_from("<Q", buf, off + 4)
        off += 12
        for _ in range(n):
            off = skip_value(buf, off, et)
        return off
    return off + SZ[t]


def parse(path):
    buf = open(path, "rb").read()
    if buf[:4] != b"GGUF":
        raise SystemExit("not a GGUF file: %s" % path)
    version, n_tensors, n_kv = struct.unpack_from("<IQQ", buf, 4)
    off = 24

    kv_bytes = []
    alignment = 32
    for _ in range(n_kv):
        start = off
        key, off = read_string(buf, off)
        (t,) = struct.unpack_from("<I", buf, off)
        off += 4
        if key == "general.alignment" and t == 4:
            (alignment,) = struct.unpack_from("<I", buf, off)
        off = skip_value(buf, off, t)
        kv_bytes.append(buf[start:off])

    tensors = {}
    for _ in range(n_tensors):
        name, off = read_string(buf, off)
        (nd,) = struct.unpack_from("<I", buf, off)
        off += 4
        dims = struct.unpack_from("<%dQ" % nd, buf, off)
        off += 8 * nd
        (tt,) = struct.unpack_from("<I", buf, off)
        off += 4 + 8
        tensors[name] = dims

    return kv_bytes, alignment, tensors


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        return 2
    base_path, src_path, dst_path = sys.argv[1:4]

    kv_bytes, alignment, base_tensors = parse(base_path)
    src_kv, src_align, _ = parse(src_path)
    if "token_embd.weight" not in base_tensors:
        raise SystemExit("base model has no token_embd.weight tensor")
    n_embd, n_vocab = base_tensors["token_embd.weight"][:2]
    alignment = src_align

    # token_embd check in llama-adapter.cpp: a->ne[1] == n_vocab, b->ne[1] == n_embd
    pair = [
        ("token_embd.weight.lora_a", (RANK, n_vocab)),
        ("token_embd.weight.lora_b", (RANK, n_embd)),
    ]

    out = bytearray(b"GGUF")
    out += struct.pack("<IQQ", 3, len(pair), len(src_kv))
    for raw in src_kv:
        out += raw

    offsets = []
    data = bytearray()
    for _, dims in pair:
        while len(data) % 32:
            data += b"\0"
        offsets.append(len(data))
        data += b"\0" * (dims[0] * dims[1] * 2)

    for (name, dims), off in zip(pair, offsets):
        raw = name.encode("utf-8")
        out += struct.pack("<Q", len(raw)) + raw
        out += struct.pack("<I", len(dims))
        out += struct.pack("<%dQ" % len(dims), *dims)
        out += struct.pack("<I", GGML_TYPE_F16) + struct.pack("<Q", off)

    while len(out) % alignment:
        out += b"\0"
    out += data
    open(dst_path, "wb").write(out)
    print("wrote %s (%d bytes): token_embd lora_a=[%d,%d] lora_b=[%d,%d]"
          % (dst_path, len(out), RANK, n_vocab, RANK, n_embd))
    return 0


if __name__ == "__main__":
    sys.exit(main())

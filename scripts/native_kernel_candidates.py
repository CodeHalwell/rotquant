"""Opt-in kernel identities; no automatic runtime-default promotion."""

W5_TILES = {"w5s8": 4, "w5s8-tile8": 8, "w5s8-tile16": 16}
DECODE_KERNELS = ("decode4", *W5_TILES)
TILED_KERNELS = ("tiled4", *DECODE_KERNELS)

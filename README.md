# Day 40 — Positional Encoding

Fixed sinusoidal position signal, injected into token embeddings
so a transformer can distinguish order.

## Core Function
PositionEncoding(max_length, dim) → (max_length, dim)

No parameters. Computed from a formula, not learned.

## The Formula
    PE(pos, 2i)   = sin(pos / 10000^(2i/d))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

d/2 frequencies, geometrically spaced from 1 to 10000.
Each frequency fills two columns — sine and cosine.

## Visualizations
- Heatmap: frequency gradient from fine checkerboard to broad bands
- Line plot: dim 0 cycles ~10× in 64 positions, dim 60 is near-flat

## Key Insight
    sin(p + k) = sin(p)cos(k) + cos(p)sin(k)

The encoding at p+k is a linear function of the encoding at p,
with coefficients that depend only on the offset. Relative position
is built into the function itself — and this is why both sine and
cosine are needed rather than one.

## Run
python demo.py
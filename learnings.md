# Learnings — Day 40

## Why Transformers Need This and RNNs Don't
An RNN inserts tokens on a timeline — t1, t2, t3 — so position
is implicit in the order of processing. A transformer processes
every token at once. It sees a set, not a sequence, and has no
idea which token sits where.

This is not a flaw being fixed. It is the price paid for
parallelism, and positional encoding buys the ordering back cheaply.

## Why Not Just Add 1, 2, 3
Two problems.

Magnitude: embeddings sit roughly in [-1, 1]. Adding 500 at
position 500 erases the token identity entirely — every token at
that position looks nearly identical.

Relative position: language cares about offsets — the adjective
two words before the noun. Hardcoded absolute indices destroy that.

## Relative Position Is the Deep Property
    sin(p + k) = sin(p)cos(k) + cos(p)sin(k)

The encoding at p+k is a linear combination of the encoding at p,
with coefficients depending only on the offset k — not on p.

"Move forward 3 positions" is the same fixed transformation at
position 5 and at position 500. The model learns it once and
applies it everywhere.

And this is exactly why both functions are needed: the right side
requires sin(p) AND cos(p). With sine alone the combination cannot
be formed. The pairing is structural, not decorative.

## Why Many Frequencies
One sine wave repeats. With period 2π it cycles every ~6 positions,
making positions 0 and 6 indistinguishable.

Many frequencies solve this the way a binary counter does: fast
columns distinguish adjacent positions, slow columns distinguish
distant ones. Together, a unique signature per position.

## Add, Not Concatenate
Concatenating a d-dim position signal doubles d_model, which doubles
the input dimension of every downstream matrix — more parameters,
more compute, for no gain.

Addition keeps d_model fixed and costs nothing. The model learns to
use some directions in the space for semantics and others for
position; they coexist rather than collide.

Note: this makes "cat at position 2" differ from "cat at position 7".
It does NOT make "bank" near river differ from "bank" near money —
that is contextual embedding, produced by attention layers mixing
tokens, not by positional encoding.

## What the Visualization Shows
Heatmap (100 × 128): tight checkerboard on the left fanning into
broad vertical bands on the right. The fan shape IS the geometric
frequency spacing, drawn.

Line plot: dim 0 completes ~10 cycles in 64 positions; dim 60 is
nearly a straight line. Position 0 is always all zeros and ones,
since sin(0)=0 and cos(0)=1 at every frequency.

## Not Universal
The original Transformer used this fixed sine scheme. BERT and GPT
switched to learned positional embeddings — simple lookup tables,
slightly better in-range, but hard-capped at the trained length
(BERT's 512-token limit is a table-size limit, not a compute one).
Most 2023+ models use RoPE, which rotates query and key vectors by
a position-dependent angle inside attention, so relative position
falls out of the geometry.

Sine is implemented here because it is what the Attention Is All
You Need paper uses, and the reasoning behind it is what RoPE builds on.

## Mechanics Worth Remembering
- torch.zeros rather than torch.empty for the table: a filling bug
  leaves a visible dead column instead of plausible garbage
- Broadcasting for +, -, *, / : align right, pad left with 1s
- Matmul is different: last two axes must satisfy the inner-dimension
  rule exactly; only the leading batch axes broadcast
- Slicing start:stop:step — 0::2 and 1::2 give the even and odd columns
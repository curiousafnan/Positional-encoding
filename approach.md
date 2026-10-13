# Approach — Day 40

## Requirements the encoding must satisfy
1. Bounded — position 500 must not swamp the embedding values
2. Relative offsets expressible — language cares about "two words
   before the noun," not "absolute position 47"
3. Extrapolates — a sequence longer than anything seen in training
   must still produce sensible values

Sine and cosine satisfy all three.

## The formula
    PE(pos, 2i)   = sin(pos / 10000^(2i/d))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

pos — the position (row)
i   — the column pair index, 0 to d/2 - 1
d   — d_model

Each pair of columns shares one frequency: sine fills the even
column, cosine fills the odd one. d/2 frequencies → d columns.

## Why geometric frequency spacing
The divisor runs from 1 (i=0) to 10000 (i = d/2 - 1).

Linear spacing would put almost every frequency in the slow range,
leaving no resolution for adjacent positions. Geometric spacing —
equal ratios between consecutive frequencies — gives the same
resolution in the 1-10 range as in the 1000-10000 range.

10000^(2i/d) walks the exponent from 0 to 1 in d/2 even steps.

## Implementation
Computed as an exponential rather than a raw power, for numerical
stability, with the sign folded in so positions are multiplied
rather than divided:

    exp(arange(0, d, 2) * (-log(10000)/d))    shape (d/2,)

Then broadcasting (max_len, 1) against (d/2,) gives the full
(max_len, d/2) angle table in one operation. Strided assignment
interleaves sine and cosine into even and odd columns.

## Shapes (max_len=16, d_model=128)
    pos_tensor   (16, 128)   the table being filled
    max_pos      (16, 1)     positions, as a column
    fun          (64,)       64 frequencies
    angle        (16, 64)    broadcast product
    [:, 0::2]    (16, 64)    even columns ← sin
    [:, 1::2]    (16, 64)    odd columns  ← cos
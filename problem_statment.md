# Day 40 — Positional Encoding

## Problem
RNNs and LSTMs get position for free: token 5 is processed after
tokens 1-4, so the hidden state inherently encodes "I am at step 5."
Transformers remove the recurrent loop to gain parallelism — and lose
that ordering. Attention is permutation-equivariant: shuffle the
inputs and the outputs shuffle identically. "dog bites man" and
"man bites dog" become indistinguishable.

Position must be injected explicitly.

## Core Questions
- Why does removing recurrence destroy order information?
- Why not simply add the position number to the embedding?
- What makes relative position expressible?
- Why many frequencies rather than one?
- Why add rather than concatenate?

## Requirements
- PositionEncoding(max_length, dim) → (max_length, dim) tensor
- Heatmap visualization of the full table
- Line plot of individual dimensions across position
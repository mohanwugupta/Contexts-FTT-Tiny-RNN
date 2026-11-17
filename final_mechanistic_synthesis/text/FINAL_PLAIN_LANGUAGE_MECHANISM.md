# Plain-Language Task-Space Mechanism

## What is the network doing?

The network learns two clearly different internal solutions for the same motor sequence: one for context A
and one for context B. Those distinct solutions are useful because context matters. The problem appears when
the cue is halfway between A and B. A High-scale network does not automatically produce a sensible halfway
prediction; its baseline Mixed loss was 2.25.

## What property fixes it without erasing the A/B split?

Only part of the hidden state can influence the next action. When that task-relevant part travels along a
sensible route between the familiar A and B solutions, the network handles the new Mixed cue better. The rest
of the hidden state can remain strongly split and nonlinear. This is why improved transfer does not require
collapsing A and B into the same representation.

## What does the wrong-path control establish?

We pushed matched networks by comparable amounts in a deliberately wrong direction. Those movements did not
provide the same benefit: wrong-path Mixed loss exceeded correct-path loss by 0.33 on average.
The result therefore concerns the direction and organization of representational change, not merely its size.

## Why is this relevant to humans, and what are we not claiming?

People also need to retain context-specific skills while responding sensibly when context is ambiguous. The
model supplies a computational explanation for how both demands can coexist. It does not show that humans
literally encode the same vectors, that this is the only possible solution, or that every trained network can
be repaired this way.

## Minimal formal statement

Let centered output logits define task state:

\[
z_k(c)=C\,[W_{out}h_k(c)+b],\qquad C=I-\frac{1}{m}\mathbf 1\mathbf 1^\top.
\]

The compositional target between A and B is:

\[
z_k^*(\lambda)=(1-\lambda)z_k(A)+\lambda z_k(B).
\]

Task-space error is:

\[
E_{task}(\lambda)=\frac{1}{7}\sum_{k=1}^7\|z_k(c_\lambda)-z_k^*(\lambda)\|_2^2.
\]

The empirical result is that reducing this error can improve held-out Mixed behavior while familiar A/B
separation stays approximately fixed.

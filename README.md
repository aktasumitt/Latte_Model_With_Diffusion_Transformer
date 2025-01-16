# Latte Model with Diffussion Transformer

The LATTE model aims to compare systems recently introduced in the field of video generation and identify the one that delivers the best results.
- Based on systems used across different categories, it has been demonstrated that the best-performing system involves:
    Sequential use of temporal and spatial transformers,
    Utilizing uniform patch embedding and absolute positional encoding,
    Applying conditioning through the s-AdaLN (adaptive layer normalization) method after layer norms.
These approaches have been shown to yield the most optimal results.

Article: [Latte Model](https://arxiv.org/abs/2401.03048)
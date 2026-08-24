# Dataset notice

The included subset is derived from
[`Anthropic/llm_global_opinions`](https://huggingface.co/datasets/Anthropic/llm_global_opinions),
which is distributed under the Creative Commons Attribution-NonCommercial-ShareAlike
4.0 license (CC BY-NC-SA 4.0).

The source dataset accompanies:

> Esin Durmus, Karina Nguyen, Thomas I. Liao, Nicholas Schiefer, Amanda Askell,
> Anton Bakhtin, Carol Chen, Zac Hatfield-Dodds, Danny Hernandez, Nicholas
> Joseph, Liane Lovitt, Sam McCandlish, Orowa Sikder, Alex Tamkin, Janel
> Thamkul, Jared Kaplan, Jack Clark, and Deep Ganguli. “Towards Measuring the
> Representation of Subjective Global Opinions in Language Models.” 2023.
> arXiv:2306.16388.

This package filters the source data to questions containing at least one valid
Australia, New Zealand, or India distribution and removes distributions for all
other countries. It does not rewrite the source questions or answer options.

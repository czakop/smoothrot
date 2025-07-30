# SmoothRot

Combining Channel-Wise Scaling and Rotation for Quantization-Friendly LLMs

## Abstract

We present SmoothRot, a novel post-training quantization technique to enhance the efficiency of 4-bit quantization in Large Language Models (LLMs). SmoothRot addresses the critical challenge of massive activation outliers, by integrating channel-wise scaling with Hadamard transformations. Our technique effectively transforms extreme outliers into quantization-friendly activations, significantly improving quantization accuracy. Experiments conducted on popular LLMs (LLaMA2 7B, LLaMA3.1 8B, and Mistral 7B) demonstrate that SmoothRot consistently reduces the performance gap between quantized and FP16 models by approximately 10-30% across language generation and zero-shot reasoning tasks, without introducing additional inference latency. 

## Citation

```bibtex
@article{czako2025smoothrot,
  title={SmoothRot: Combining Channel-Wise Scaling and Rotation for Quantization-Friendly LLMs},
  author={Czakó, Patrik and Kertész, Gábor and Szénási, Sándor},
  journal={arXiv preprint arXiv:2506.05413},
  year={2025}
}
```

# deep-learning-blueprint

"True understanding comes from building the engine, not just driving the car." This repository documents my systematic exploration of advanced DL concepts – implementing every component from scratch without high-level APIs to force fundamental understanding.


## The Implementation Roadmap

A checklist of concepts and models to implement.

### 🧠 Core Architectural Concepts & Building Blocks

  - [x] **Normalization Layers**
      - [x] Batch Normalization : ([NormalizationLayers.ipynb](NormalizationLayers.ipynb#BatchNorm))
      - [x] Layer Normalization : ([NormalizationLayers.ipynb](NormalizationLayers.ipynb#LayerNorm))
      - [x] Instance Normalization : ([NormalizationLayers.ipynb](NormalizationLayers.ipynb#InstanceNorm))
      - [x] Group Normalization : ([NormalizationLayers.ipynb](NormalizationLayers.ipynb#GroupNorm))
  - [ ] **Activation Functions**
      - [x] **ReLU Variants**: Leaky ReLU, Parametric ReLU (PReLU), Exponential Linear Unit (ELU)
      - [ ] **Gated Activations**: Gated Linear Unit (GLU), SwiGLU, GeGLU
      - [ ] **Advanced**: GELU (Gaussian Error Linear Unit), Swish / SiLU, Mish
      - [ ] **Efficient**: Hard Sigmoid, Hard Swish
  - [ ] **Convolutional Variants**
      - [ ] Dilated (Atrous) Convolution
      - [ ] Depthwise Separable Convolution
      - [ ] Deformable Convolution
  - [x] **Attention Mechanisms (beyond basic self-attention)**
      - [x] **Self-Attention**: Multi-head attention mechanism with causal masking
      - [x] **Multi-Head Latent Attention (MLA)**: DeepSeek-style MLA ([SimpleDeepseek.ipynb](SimpleDeepseek.ipynb))
      - [ ] Cross-Attention
      - [ ] FlashAttention (I/O-aware implementation)
      - [ ] Sparse/Linear Attention (e.g., in Longformer, Performer)

### 🎨 Generative Models

  - [ ] **Variational Autoencoders (VAEs)**
      - [ ] Core VAE (Reparameterization trick, ELBO loss: $\\log p(x) \\ge \\mathbb{E}*{q(z|x)}[\\log p(x|z)] - D*{KL}(q(z|x) || p(z))$)
      - [ ] Conditional VAE (CVAE)
      - [ ] Vector Quantized VAE (VQ-VAE)
  - [ ] **Generative Adversarial Networks (GANs)**
      - [ ] Wasserstein GAN (WGAN-GP)
      - [ ] CycleGAN
      - [ ] StyleGAN
  - [ ] **Diffusion Models**
      - [ ] Denoising Diffusion Probabilistic Models (DDPM)
      - [ ] Latent Diffusion Models (LDM)
  - [ ] **Autoregressive Models**
      - [ ] PixelCNN
  - [ ] **Normalizing Flows**
      - [ ] RealNVP or GLOW

### 💬 Advanced NLP & Large Language Models

  - [x] **Core Transformer & GPT Implementation**
      - [x] **[SimpleGPT](SimpleGPT.ipynb)**: Complete GPT implementation from scratch with multi-head attention, transformer blocks, and autoregressive text generation
      - [x] **[Multi-Head Attention](SimpleGPT.ipynb#multi-head-attention)**: Self-attention mechanism with causal masking
      - [x] **[Transformer Architecture](SimpleGPT.ipynb#transformer-blocks)**: Positional embeddings, transformer blocks, feed-forward networks
      - [x] **[Rotary Position Embeddings (RoPE)](SimpleDeepseek.ipynb)**: Rotary positional encoding (see "Model Creation (MLA with RoPE positional Encoding)")
  - [ ] **Efficient Transformers**
      - [ ] Longformer / BigBird
      - [ ] Reformer
  - [ ] **Modern LLM Architectures**
      - [ ] Mixture of Experts (MoE) layer
      - [ ] State Space Models (**Mamba**)
  - [ ] **Parameter-Efficient Fine-Tuning (PEFT)**
      - [ ] Low-Rank Adaptation (**LoRA**)
  - [ ] **LLM Application Paradigms**
      - [ ] Retrieval-Augmented Generation (**RAG**) System

### 👁️ Advanced Computer Vision

  - [ ] **Object Detection**
      - [ ] YOLO (You Only Look Once)
      - [ ] DETR (DEtection TRansformer)
  - [ ] **Segmentation**
      - [ ] U-Net
      - [ ] Vision Transformer for segmentation
  - [ ] **Transformers in Vision**
      - [ ] Vision Transformer (**ViT**)
      - [ ] Swin Transformer
  - [ ] **3D Vision & Scene Representation**
      - [ ] Neural Radiance Fields (**NeRF**)
      - [ ] PointNet / PointNet++

### 🕸️ Graph-based & Geometric Deep Learning

  - [ ] **Graph Neural Networks (GNNs)**
      - [ ] Graph Convolutional Networks (GCN)
      - [ ] GraphSAGE
      - [ ] Graph Attention Networks (GAT)

### 🤖 Deep Reinforcement Learning

  - [ ] **Value-Based Methods**
      - [ ] Double Dueling DQN
  - [ ] **Advanced Actor-Critic Methods**
      - [ ] Proximal Policy Optimization (**PPO**)
      - [ ] Soft Actor-Critic (**SAC**)

### 📚 Unsupervised, Self-Supervised & Multi-Modal

  - [ ] **Contrastive Learning**
      - [ ] SimCLR
      - [ ] MoCo
  - [ ] **Masked Modeling**
      - [ ] Masked Autoencoders (**MAE**) for vision
  - [ ] **Multi-Modal Models**
      - [ ] CLIP (Replicating the training approach on a smaller scale)

### ⚙️ Model Optimization & Efficiency

  - [ ] **Knowledge Distillation**
  - [ ] **Network Pruning** (e.g., Lottery Ticket Hypothesis)
  - [ ] **Quantization** (Post-Training and Quantization-Aware)

### 🔬 Theoretical & Mathematical Foundations

  - [ ] **Bayesian Deep Learning**
      - [ ] Uncertainty Estimation via MC Dropout
  - [ ] **Advanced Optimizers**
      - [ ] **Adaptive Learning Rate**: AdaGrad, RMSprop
      - [ ] **Adam Variants**: AdamW (Decoupled Weight Decay), RAdam (Rectified Adam)
      - [ ] **Recent Developments**: Lion (Evo**L**ved S**i**gn M**o**me**n**tum), Lookahead
      - [ ] **Regularization-based**: Sharpness-Aware Minimization (SAM)

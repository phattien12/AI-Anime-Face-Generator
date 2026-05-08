<div align="center">

<h1>AI Anime Face Generator using DCGAN, Self-Attention GAN, Style Mapping, and WGAN-GP</h1>

<p>
A complete deep learning project for generating high-quality anime faces using multiple GAN architectures trained on the Anime Faces dataset with PyTorch and Google Colab.
</p>

<img src="https://img.shields.io/badge/Python-3.10-blue?logo=python">
<img src="https://img.shields.io/badge/PyTorch-DeepLearning-red?logo=pytorch">
<img src="https://img.shields.io/badge/GPU-NVIDIA%20T4-green?logo=nvidia">
<img src="https://img.shields.io/badge/Platform-Google%20Colab-orange?logo=googlecolab">
<img src="https://img.shields.io/badge/Project-GAN-purple">

</div>

---

# 📌 Project Overview

This project demonstrates the complete pipeline for building an advanced AI Anime Face Generator using several Generative Adversarial Network (GAN) architectures.

The notebook starts with a standard DCGAN implementation and progressively improves the image quality by introducing:

<ul>
<li>Better weight initialization</li>
<li>Label smoothing</li>
<li>Self-Attention mechanism</li>
<li>Style Mapping Network</li>
<li>Spectral Normalization</li>
<li>WGAN-GP (Wasserstein GAN with Gradient Penalty)</li>
<li>EMA (Exponential Moving Average)</li>
<li>Mixed Precision Training (AMP)</li>
</ul>

The final model is capable of generating realistic anime-style character faces from random latent vectors.

---

# 🧠 Main Objectives

<ul>
<li>Understand how GANs work internally</li>
<li>Train a GAN model from scratch using PyTorch</li>
<li>Generate realistic anime face images</li>
<li>Experiment with advanced GAN stabilization techniques</li>
<li>Compare different GAN architectures</li>
<li>Learn practical GPU optimization techniques</li>
<li>Deploy a complete GAN training pipeline on Google Colab</li>
</ul>

---

# 🏗️ Architectures Used

# 1️⃣ DCGAN (Deep Convolutional GAN)

The first model implemented in this project is a classic DCGAN.

DCGAN consists of two neural networks:

<ul>
<li><b>Generator (G)</b> → Generates fake anime faces from random noise.</li>
<li><b>Discriminator (D)</b> → Distinguishes between real and fake images.</li>
</ul>

The Generator uses:

<ul>
<li>ConvTranspose2D layers</li>
<li>Batch Normalization</li>
<li>ReLU activations</li>
<li>Tanh output activation</li>
</ul>

The Discriminator uses:

<ul>
<li>Conv2D layers</li>
<li>LeakyReLU activations</li>
<li>Batch Normalization</li>
<li>Sigmoid output activation</li>
</ul>

This architecture creates 64x64 RGB anime faces.

---

# 2️⃣ Improved DCGAN

The project then improves the baseline DCGAN using:

<ul>
<li>Custom weight initialization</li>
<li>Label smoothing</li>
<li>Better GAN balancing</li>
<li>Improved stability</li>
</ul>

Label smoothing changes:

<ul>
<li>Real labels from 1.0 → 0.9</li>
<li>Fake labels remain 0.0</li>
</ul>

This prevents the discriminator from becoming too confident.

---

# 3️⃣ Self-Attention GAN (SAGAN)

The project introduces Self-Attention layers to improve global feature relationships.

Self-Attention allows the model to:

<ul>
<li>Focus on important image regions</li>
<li>Learn long-range dependencies</li>
<li>Improve facial structure consistency</li>
<li>Generate better eye and hair details</li>
</ul>

Attention layers were added inside:

<ul>
<li>Generator</li>
<li>Discriminator</li>
</ul>

This significantly improves anime face quality.

---

# 4️⃣ Style Mapping Network

Inspired by StyleGAN architecture.

Instead of directly feeding random latent vectors into the generator:

<ul>
<li>Random vector z is first transformed by a Mapping Network</li>
<li>The mapping network learns a better latent representation</li>
<li>This improves feature disentanglement</li>
<li>Training becomes more stable</li>
</ul>

The mapping network contains several fully connected layers.

---

# 5️⃣ Spectral Normalization

Spectral Normalization is applied to the discriminator.

Benefits:

<ul>
<li>Controls discriminator gradients</li>
<li>Prevents exploding gradients</li>
<li>Improves GAN stability</li>
<li>Reduces mode collapse</li>
</ul>

This is extremely important for advanced GAN training.

---

# 6️⃣ WGAN-GP (Wasserstein GAN with Gradient Penalty)

The final and most advanced architecture uses:

<ul>
<li>Wasserstein loss</li>
<li>Gradient Penalty</li>
<li>Self-Attention</li>
<li>EMA Generator</li>
<li>Mixed Precision Training</li>
</ul>

WGAN-GP improves:

<ul>
<li>Training stability</li>
<li>Gradient flow</li>
<li>Image realism</li>
<li>GAN convergence</li>
</ul>

Unlike standard GANs:

<ul>
<li>Discriminator becomes Critic</li>
<li>No Sigmoid output</li>
<li>Uses Wasserstein distance</li>
</ul>

Gradient penalty enforces Lipschitz continuity.

---

# ⚡ EMA (Exponential Moving Average)

The project also uses EMA for the generator.

EMA creates a smoothed version of the generator weights.

Benefits:

<ul>
<li>Cleaner generated images</li>
<li>Less noisy outputs</li>
<li>More stable inference</li>
<li>Higher visual quality</li>
</ul>

EMA update equation:

<pre>
ema = decay * ema + (1 - decay) * current
</pre>

EMA decay used:

<pre>
0.999
</pre>

---

# 🚀 Mixed Precision Training (AMP)

Automatic Mixed Precision was used using:

<ul>
<li>autocast()</li>
<li>GradScaler()</li>
</ul>

Benefits:

<ul>
<li>Faster training</li>
<li>Lower GPU memory usage</li>
<li>Better GPU utilization</li>
<li>Suitable for Colab T4 GPU</li>
</ul>

---

# 📂 Dataset

Dataset used:

<b>Anime Faces Dataset</b>

Kaggle Dataset:

<pre>
https://www.kaggle.com/datasets/soumikrakshit/anime-faces
</pre>

Dataset size:

<pre>
21,551 anime face images
</pre>

Image preprocessing:

<ul>
<li>Resize to 64x64</li>
<li>Convert to tensor</li>
<li>Normalize to [-1, 1]</li>
</ul>

Normalization:

<pre>
Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
</pre>

---

# 🖥️ Training Environment

Training was performed on:

<table>
<tr>
<td><b>Platform</b></td>
<td>Google Colab</td>
</tr>
<tr>
<td><b>GPU</b></td>
<td>NVIDIA Tesla T4</td>
</tr>
<tr>
<td><b>CUDA Version</b></td>
<td>13.0</td>
</tr>
<tr>
<td><b>VRAM</b></td>
<td>15GB</td>
</tr>
</table>

---

# 📦 Required Libraries

Install dependencies:

```bash
pip install torch torchvision matplotlib tqdm pillow kaggle
```

Main libraries used:

<ul>
<li>PyTorch</li>
<li>Torchvision</li>
<li>Matplotlib</li>
<li>TQDM</li>
<li>Pillow</li>
<li>Kaggle API</li>
</ul>

---

# 📥 Dataset Download

```python
!kaggle datasets download -d soumikrakshit/anime-faces
```

Unzip dataset:

```python
!unzip -q anime-faces.zip
```

---

# 📁 Project Structure

```text
AI-Anime-Face-Generator/
│
├── anime_dataset/
│   └── anime/
│
├── generated_samples/
├── better_samples/
├── stylegan_samples/
├── wgan_gp_samples/
│
├── anime_generator.pth
├── anime_discriminator.pth
│
├── stylegan_generator.pth
├── stylegan_discriminator.pth
├── stylegan_mapping_network.pth
│
├── wgan_ema_generator.pth
├── wgan_discriminator.pth
├── wgan_mapping_network.pth
│
├── AI_anime_face_generator.ipynb
│
└── README.md
```

---

# 🔥 Hyperparameters

<table>
<tr>
<td><b>Parameter</b></td>
<td><b>Value</b></td>
</tr>
<tr>
<td>Image Size</td>
<td>64x64</td>
</tr>
<tr>
<td>Batch Size</td>
<td>128</td>
</tr>
<tr>
<td>Latent Dimension</td>
<td>100</td>
</tr>
<tr>
<td>Epochs</td>
<td>50</td>
</tr>
<tr>
<td>Learning Rate</td>
<td>0.0002 / 0.0001</td>
</tr>
<tr>
<td>Beta1</td>
<td>0.5</td>
</tr>
<tr>
<td>EMA Decay</td>
<td>0.999</td>
</tr>
<tr>
<td>Gradient Penalty</td>
<td>10</td>
</tr>
<tr>
<td>N Critic</td>
<td>5</td>
</tr>
</table>

---

# 🧩 Generator Pipeline

The Generator workflow:

```text
Random Noise Vector
        ↓
Mapping Network
        ↓
Reshape Latent Tensor
        ↓
ConvTranspose Blocks
        ↓
Self-Attention Layers
        ↓
Generated Anime Face
```

---

# 🧩 Discriminator Pipeline

The Discriminator workflow:

```text
Input Image
     ↓
Convolution Blocks
     ↓
Self-Attention Layers
     ↓
Spectral Normalization
     ↓
Real/Fake Prediction
```

---

# 🏋️ Training Process

During training:

<ol>
<li>Real images are loaded from dataset</li>
<li>Random latent vectors are sampled</li>
<li>Generator creates fake anime faces</li>
<li>Discriminator evaluates real and fake images</li>
<li>Losses are computed</li>
<li>Backpropagation updates weights</li>
<li>EMA weights are updated</li>
<li>Sample images are saved after each epoch</li>
</ol>

---

# 🖼️ Generated Samples

The notebook automatically saves generated images after every epoch.

Example:

```python
save_image(
    fake,
    f"{sample_dir}/epoch_{epoch+1:03d}.png",
    normalize=True,
    nrow=8
)
```

This allows visual monitoring of GAN learning progress.

---

# 📈 Training Observations

# DCGAN

<ul>
<li>Initially blurry outputs</li>
<li>Slow convergence</li>
<li>Some mode collapse issues</li>
<li>Basic anime face structure learned</li>
</ul>

# Improved DCGAN

<ul>
<li>More stable training</li>
<li>Better eye details</li>
<li>Cleaner facial features</li>
</ul>

# Self-Attention GAN

<ul>
<li>Sharper faces</li>
<li>Improved global consistency</li>
<li>Better hair structures</li>
</ul>

# WGAN-GP + EMA

<ul>
<li>Best image quality</li>
<li>Most stable training</li>
<li>Reduced artifacts</li>
<li>Most realistic anime faces</li>
</ul>

---

# 💾 Model Saving

Generator:

```python
torch.save(G.state_dict(), "anime_generator.pth")
```

Discriminator:

```python
torch.save(D.state_dict(), "anime_discriminator.pth")
```

EMA Generator:

```python
torch.save(ema_G.state_dict(), "wgan_ema_generator.pth")
```

---

# ▶️ Inference Example

Generate new anime faces:

```python
G.eval()

noise = torch.randn(
    64,
    latent_dim,
    1,
    1,
    device=device
)

with torch.no_grad():
    generated = G(noise).cpu()
```

---

# 📊 Features Implemented

<ul>
<li>DCGAN</li>
<li>Improved DCGAN</li>
<li>Self-Attention GAN</li>
<li>Style Mapping Network</li>
<li>Spectral Normalization</li>
<li>WGAN-GP</li>
<li>Gradient Penalty</li>
<li>EMA Generator</li>
<li>Mixed Precision Training</li>
<li>PyTorch AMP</li>
<li>Sample Saving</li>
<li>Model Checkpointing</li>
<li>Google Colab Support</li>
</ul>

---

# ⚠️ Challenges During Training

GAN training is notoriously unstable.

Challenges encountered:

<ul>
<li>Mode collapse</li>
<li>Discriminator overpowering generator</li>
<li>Exploding gradients</li>
<li>Training instability</li>
<li>Noisy outputs</li>
</ul>

Solutions used:

<ul>
<li>Label smoothing</li>
<li>Spectral normalization</li>
<li>Gradient penalty</li>
<li>Self-attention</li>
<li>EMA smoothing</li>
<li>WGAN objective</li>
</ul>

---

# 🚀 Performance

Approximate training speeds on Tesla T4:

<table>
<tr>
<td><b>Architecture</b></td>
<td><b>Speed</b></td>
</tr>
<tr>
<td>DCGAN</td>
<td>~5 it/s</td>
</tr>
<tr>
<td>SAGAN</td>
<td>~3.8 it/s</td>
</tr>
<tr>
<td>WGAN-GP + AMP</td>
<td>~5-7 it/s</td>
</tr>
</table>

---

# 📌 Future Improvements

Possible future upgrades:

<ul>
<li>Train at 128x128 or 256x256 resolution</li>
<li>Use official StyleGAN2</li>
<li>Add progressive growing</li>
<li>Conditional GAN support</li>
<li>Anime character attribute control</li>
<li>Text-to-anime generation</li>
<li>Diffusion model integration</li>
<li>Web deployment with Gradio</li>
</ul>

---

# 🌟 Learning Outcomes

This project teaches:

<ul>
<li>GAN fundamentals</li>
<li>Generator and discriminator design</li>
<li>Advanced GAN stabilization</li>
<li>Attention mechanisms</li>
<li>Latent space mapping</li>
<li>Mixed precision optimization</li>
<li>PyTorch training workflows</li>
<li>GPU-efficient deep learning</li>
</ul>

---

# 🧪 Example Results

The final WGAN-GP + EMA model generates:

<ul>
<li>Sharp anime eyes</li>
<li>Clean facial structures</li>
<li>Consistent hairstyles</li>
<li>Better color harmony</li>
<li>Reduced visual artifacts</li>
</ul>

Compared to the baseline DCGAN, the final architecture produces significantly higher-quality images.

---

# 📝 Notes

<ul>
<li>The project was fully trained on Google Colab.</li>
<li>The notebook is optimized for Tesla T4 GPU.</li>
<li>Mixed Precision Training significantly reduces VRAM usage.</li>
<li>Training time depends on GPU availability.</li>
<li>Generated images improve progressively with epochs.</li>
</ul>

---

# 👨‍💻 Author

<div>
<b>AI Anime Face Generator Project</b><br>
Built with PyTorch, GANs, Self-Attention, WGAN-GP, and Style Mapping.<br>

Focused on practical deep learning experimentation and anime image synthesis.

</div>

---

# 📜 License

This project is intended for:

<ul>
<li>Research</li>
<li>Educational purposes</li>
<li>Deep learning experiments</li>
<li>GAN learning demonstrations</li>
</ul>

Please verify dataset licensing before commercial usage.

---

# ❤️ Final Summary

This project evolves from a simple DCGAN into a highly advanced anime face generation system using multiple modern GAN stabilization techniques.

The notebook demonstrates a full deep learning workflow including:

<ul>
<li>Dataset preparation</li>
<li>GAN architecture design</li>
<li>Training optimization</li>
<li>Advanced stabilization methods</li>
<li>Self-attention mechanisms</li>
<li>WGAN-GP implementation</li>
<li>EMA smoothing</li>
<li>Mixed precision acceleration</li>
<li>Model saving and inference</li>
</ul>

The final WGAN-GP + EMA architecture achieves the best balance between stability, realism, and training efficiency.

This project serves as a strong foundation for advanced generative AI research and anime image synthesis.


# Unsupervised Image Compression via Cycle Consistency

> Proposed a direct deep learning based image compression method via cycle consistency, where the neural network consists of an encoder and a symmetrical decoder. Both reconstruction loss and sparsity loss objectives are combined as the loss for the training process. Achieved a PSNR of 21 on validation set, and compression ratio of 5.33. The fully trained model is capable of compressing images for moderate quality with a very short runtime.


<a><img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ggd8xns01bj31sz0u0jvn.jpg" title="Ipic is tring to help!" alt="Ipic is tring to help!"></a>

- Baby IPic is trying to help here!


**Files Included**

- proposal version 1.0
- proposal version 2.0
- Cyclic autoencoder using PyTorch
- Cyclic autoencoder using Keras ( to be refined )


## Table of Contents (Optional)


- [Introduction](#Introduction)
- [Contributing](#contributing)
- [Team](#team)
- [License](#license)

---

## Introduction

- We think the problem of image compression of quality loss is a dual task of image super-resolution. Inspired by CycleGAN (Jun-Yan Zhu & Efros, 2017), we propose a direct and simple deep learning based image compres- sion algorithm via cycle consistency (Figure 1) where the "symmetrical" decoder reconstructs the compressed image into source image if we find a encoder which can be a deep neural network for compression. The encoder-decoder is successful whenever the fully trained decoder reconstructs the compressed code into source image. Our proposed al- gorithm is also inspired by (Caglar Aytekin & Hannuksela, 24 May 2019) with some additional design and details in Methodology.

<a><img src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ggd8f3yayyj31hf0u0njs.jpg" title="Methodology" alt="Methodology"></a>

- And here's an example of how we reconstruct images:

<a><img style="float: left;" src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ggd9ay2jhxj30b40b2mya.jpg" title="original" alt="original"> <img style="float: left;" src="https://tva1.sinaimg.cn/large/007S8ZIlgy1ggd9azs22ej30b60b4jsm.jpg" title="reconstructed" alt="reconstructed"> </a>

---

## Contributing

> New things start here...

### Step 1

- **Option 1**
    - 🍴 Fork this repo!

- **Option 2**
    - 👯 Clone this repo to your local machine

### Step 2

- **HACK AWAY!** 🔨🔨🔨

### Step 3

- 🔃 Create a new pull request and we will get back to you!

---

## Team

> Our contributors

- Fei Gu, filed all needed references and introduced methodology in the representation .
- Muwei He (Orinsa), implemented part of the algorithm, evaluated the model on datasets and represented testing results.
- Bradley He, proposed the methodology, and implemented most part of the algorithm.

---

## License

- Copyright 2020 © <a href="https://github.com/OrinsaCinder" target="_blank">Orinsa Cinder</a>.
- May contact Orinsa for more information and access to our presentation video.
---

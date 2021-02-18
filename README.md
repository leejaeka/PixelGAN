# PixelGAN
 pixel flowers generation using WGAN-GP. The project is best described with pictures below.

![main](/output/1.png)
![main](/output/2.png)
![main](/output/3.png)
![main](/output/4.png)
![main](/output/9.png)
![main](/output/11.png)
![main](/output/5.png)
![main](/output/10.png)

<br>
![main](/output/iter_100.png)
![main](/output/iter_300.png)
![main](/output/iter_600.png)
![main](/output/iter_900.png)
![main](/output/iter_2000.png)
![main](/output/iter_4000.png)


## Dataset 
100 (16 x 16)pixels flower art by a user BTL games from Itch.io. [Link](https://btl-games.itch.io/pixel-art-fauna-asset-pack)

## Reference
[Generative Deep Learning](https://www.oreilly.com/library/view/generative-deep-learning/9781492041931/) by David Foster
Wasserstein GAN (Arjovsky, Chintala, and Bottou, 2017): https://arxiv.org/abs/1701.07875
Improved Training of Wasserstein GANs (Gulrajani et al., 2017): https://arxiv.org/abs/1704.00028

## Reflection
It is interesting to note that although it is obvious to our eyes which ones are fake due to the background being blurry green, it is still able to fool the discriminator. 
Note to self: For next time, try TPU so it won't run out of memory so often. 

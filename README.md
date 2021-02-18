# PixelGAN
16 by 16 pixel flowers generation using WGAN-GP. The project is best described with pictures below. All the flowers are GAN generated meaning they don't exist in real world! Note that with the generator, I can generate infinitely unique flowers. Here are some of my favorites. <br/>
[Link to Google Colab Notebook](https://colab.research.google.com/drive/1Q5RnFuy6C4dfmyuiHDeFNIpotDYohCi6?usp=sharing)

![main](/output/1.png)
![main](/output/77.png)
![main](/output/3.png)
![main](/output/2.png)
![main](/output/9.png)
![main](/output/11.png)
![main](/output/5.png)
![main](/output/10.png)
<br/>
Below are generated vs real samples at different training steps
<br/>
Iteration 100
<br/>
![hi](/output/iter_100.png)
<br/>
Iteration 300
<br/>
![main](/output/iter_300.png)
<br/>
Iteration 600
<br/>
![main](/output/iter_600.png)
<br/>
Iteration 900
<br/>
![main](/output/iter_900.png)
<br/>
Iteration 2000
<br/>
![main](/output/iter_2000.png)
<br/>
Iteration 4000
<br/>
As we go Deeper into training, we can start seeing some crazy but pretty flowers
<br/>
![main](/output/iter_4000.png)
<br/>
Plot of Criterion_Loss vs Generator_Loss (x-axis scaled by 1/300)
<br/>
![main](/output/graphy.png)

## Dataset 
100 (16 x 16)pixels flower art by a user BTL games from Itch.io. [Link](https://btl-games.itch.io/pixel-art-fauna-asset-pack)

## Reference
- [Generative Deep Learning](https://www.oreilly.com/library/view/generative-deep-learning/9781492041931/) by David Foster
- Wasserstein GAN (Arjovsky, Chintala, and Bottou, 2017): https://arxiv.org/abs/1701.07875
- Improved Training of Wasserstein GANs (Gulrajani et al., 2017): https://arxiv.org/abs/1704.00028

## Reflection
- It is interesting to note that although it is obvious to our eyes which ones are fake due to the background being blurry green, it is still able to fool the discriminator. 
- Note to self: For next time, try TPU so it won't run out of memory so often. 

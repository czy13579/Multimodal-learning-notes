## GLIDE 作用
探索使用扩散模型实现文本条件下的图像生成，并对比了两种不同的 guidance 策略，分别是通过 CLIP 引导和classifier-free 的引导。验证了 classifier-free 的方式生成的图片更真实，与提示的文本有更好的相关性。并且使用 classifier-free 的引导的 GLIDE 模型在 35 亿参数的情况下优于 120 亿参数的 DALL-E 模型。

## GLIDE 模型结构

作者训练的模型包括：

- 一个 35 亿参数量的 text-conditional 扩散模型，分辨率为 64*64
- 一个 15 亿参数量的 text-conditional 上采样扩散模型，将分辨率提升至 256x256
- 对于 CLIP guidance 模型，还额外训练了一个 64x64 noised ViT-L CLIP


## SUPIR
##### SUPIR是一个图像修复的方法。
![alt text](image.png)

* 生成模型：SDXL
*  Degradation-Robust Encoder
    * 把Low-Quality图片转换成Low-Quality的Latent Feature.
    * Fine-tune了原始SDXL的Encoder。
* New Adaptor.
  * ControlNet Structure + Network Trimming.
  * A ZeroSFT connector to control SDXL IR Model.
* Multi-modal Language Guidance
  * LLaVA把图片转换成prompt, 再将prompt输入到ControlNet部分。
  * collect textual annotations for all the training image to reinforce control.
* Negative-Quality Samples and Prompt
$$
\begin{array}{l}
z_{t-1}^{\text {pos }}=\mathcal{H}\left(z_{t}, z_{L Q}, \sigma_{t}, \operatorname{pos}\right), z_{t-1}^{\text {neg }}=\mathcal{H}\left(z_{t}, z_{L Q}, \sigma_{t}, \text { neg }\right), \\
z_{t-1}=z_{t-1}^{\text {pos }}+\lambda_{\text {cfg }} \times\left(z_{t-1}^{\text {pos }}-z_{t-1}^{\text {neg }}\right)
\end{array} 
$$ 
其中，$\lambda_{cfg}$是超参数。H是Diffusion model.
* Modify EDM Sampling to make sure IR faithfully.

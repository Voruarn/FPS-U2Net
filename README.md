# FPSU2Net, CAGEO 2024</a> </p>

- Paper: [FPS-U2Net: Combining U2Net and multi-level aggregation architecture for fire point segmentation in remote sensing images](https://www.sciencedirect.com/science/article/pii/S0098300424001110)


## Abstract

Traditional methods for fire point segmentation (FPS) in satellite remote sensing images (RSIs) overly rely on threshold judgment, which are greatly affected by factors such as regional time and show poor generalization. Besides, due to the difference between natural scene images (NSIs) and RSIs, directly apply NSIs-based deep learning methods to forest fire RSIs without any modification fails to achieve satisfactory results. To address these issues, first, we construct a Landsat8 RSI-FPS dataset covering different years, seasons and regions. Then, for the first time, we apply salient object detection (SOD) to FPS in forest fire monitoring and propose a novel network FPS-U2Net to improve the performance of FPS. FPS-U2Net is based on U2Netp (a lightweight U2Net), to make better use of the multi-level features from adjacent encoders, we propose multi-level aggregation module (MAM), which is placed between the encoder and decoder at the same stage to aggregate the adjacent multi-scale features and capture richer contextual information. To make up for the weakness of BCE loss, we employ the hybrid loss, BCE + IoU, for the training of the network, which can guide the network learn the salient information from pixel and map levels. Extensive experiments on three datasets demonstrate that our FPS-U2Net significantly outperforms the state-of-the-art semantic segmentation and SOD methods. FPS-U2Net can accurately segment fire regions and predict clear local details.

## Related Works (Forest Fire Monitoring)
[Dual backbone interaction network for burned area segmentation in optical remote sensing images ](https://github.com/Voruarn/DBINet), IEEE GRSL 2024.

[Burned Area Segmentation in Optical Remote Sensing Images Driven by U-Shaped Multistage Masked Autoencoder ](https://github.com/Voruarn/DCNet), IEEE JSTARS 2024.

[A novel salient object detection network for burned area segmentation in high-resolution remote sensing images ](https://github.com/Voruarn/PANet), ENVSOFT 2025.

[Controllable diffusion generated dataset and hybrid CNN–Mamba network for burned area segmentation ](https://github.com/Voruarn/HCM), ADVEI 2025.


```
## 📎 Citation

If you find the code helpful in your research or work, please cite the following paper(s).

@article{FANG2024105628,
    title = {FPS-U2Net: Combining U2Net and multi-level aggregation architecture for fire point segmentation in remote sensing images},
    journal = {Computers & Geosciences},
    volume = {189},
    pages = {105628},
    year = {2024},
    issn = {0098-3004},
    author = {Wei Fang and Yuxiang Fu and Victor S. Sheng},
  }
```

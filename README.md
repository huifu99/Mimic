## **Mimic**

Official PyTorch implementation for the paper:

> **Mimic: Speaking Style Disentanglement for Speech-Driven 3D Facial Animation**, ***AAAI 2024***.
>
> Hui Fu, Zeqing Wang, Ke Gong, Keze Wang, Tianshui Chen, Haojie Li, Haifeng Zeng, Wenxiong Kang
>
> <a href='https://arxiv.org/pdf/2312.10877.pdf'><img src='https://img.shields.io/badge/arXiv-2312.10877-red'></a> <a href='https://zeqing-wang.github.io/Mimic/'><img src='https://img.shields.io/badge/Project-Video-Green'></a>

<p align="center">
<img src="framework.png" width="75%"/>
</p>

>Speech-driven 3D facial animation aims to synthesize vivid facial animations that accurately synchronize with speech and match the unique speaking style. However, existing works primarily focus on achieving precise lip synchronization while neglecting to model the subject-specific speaking style, often resulting in unrealistic facial animations. To the best of our knowledge, this work makes the first attempt to explore the coupled information between the speaking style and the semantic content in facial motions. Specifically, we introduce an innovative speaking style disentanglement method, which enables arbitrary-subject speaking style encoding and leads to a more realistic synthesis of speech-driven facial animations. Subsequently, we propose a novel framework called **Mimic** to learn disentangled representations of the speaking style and content from facial motions by building two latent spaces for style and content, respectively. Moreover, to facilitate disentangled representation learning, we introduce four well-designed constraints: an auxiliary style classifier, an auxiliary inverse classifier, a content contrastive loss, and a pair of latent cycle losses, which can effectively contribute to the construction of the identity-related style space and semantic-related content space. Extensive qualitative and quantitative experiments conducted on three publicly available datasets demonstrate that our approach outperforms state-of-the-art methods and is capable of capturing diverse speaking styles for speech-driven 3D facial animation.

<p align="center">
<img src="comparisons.png" width="95%"/>
</p>

## **TODO**
- ~~Release codes and weights for inference.~~
- Release 3D-HDTF dataset.
- Release codes for training.

## **Environment**
- Ubuntu
- RTX 4090
- CUDA 11.6 (GPU with at least 24GB VRAM)
- Python 3.9
 ## **Dependencies**
- PyTorch 1.13.1
- ffmpeg
- [PyTorch3D](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) (recommend) or [MPI-IS/mesh](https://github.com/MPI-IS/mesh) for rendering

Other necessary packages:
```
pip install -r requirements.txt
```

 ## **Demo**
 We provide some demos for 3D-HDTF. Please follow the process to run the demos.

 1) Prepare data and pretrained models
    
    Clone the repository using: `git clone https://github.com/huifu99/Mimic.git` .

    Download the 3D-HDTF [data](https://drive.google.com/drive/folders/1s9FQQRAA_tqf58ThmWq3EWmFnV7syhhl?usp=drive_link) for demos and [model](https://drive.google.com/drive/folders/122oSYMiwQyzg8kvfWhp6A-JejMZGCNM8?usp=drive_link) trained using 3D-HDTF. Then put them to the root directory of Mimic.

    Prepare the [SPECTRE model trained on HDTF](https://drive.google.com/drive/folders/18YM4J4u5Tpi-JLQLh-_UwDUwSPoVZBb1?usp=drive_link) and [dependencies](https://drive.google.com/drive/folders/197z4B8GYZ9QFwzGFXBkIZtddHESlgg1K?usp=drive_link)  of SPECTRE and put it to `external/spectre/`.
    Organize the files into the following structure:
    ```
      Mimic
      │
      └─── demos
         │
         └─── wav
         │
         └─── style_ref
      │
      └─── pretrained
         │
         └───<experiment name>
            │
            └─── Epoch_x.pth
      │
      └─── external
         │
         └───spectre
            │
            └─── data
            │
            └─── pretrained
               │
               └─── HDTF_pretrained
            │
            └─── ...
      │
      └─── ...
    ```

 2) Run demos
    
    Run the following command to get the demo results (.npy file for vertices and .mp4 for videos) in `demos/results`:

    ```
    python demo.py --wav_file demos/wav/RD_Radio11_001.wav --style_ref id_002-RD_Radio11_001
    ```

    Your can change the parameters such as `--wav_file` and `--style_ref` according to your path. The process of generating style reference file will be provided soon.

 ## **Training and evaluation**


## **Acknowledgement**
We heavily borrow the code from
[CodeTalker](https://github.com/Doubiiu/CodeTalker),
[VOCA](https://github.com/TimoBolkart/voca) and [SPECTRE](https://github.com/filby89/spectre). Thanks
for sharing their code. Our 3D-HDTF dataset is based on [HDTF](https://github.com/MRzzm/HDTF). Third-party packages are owned by their respective authors and must be used under their respective licenses.

## **Citation**

If you find the code useful for your work, please star this repo and consider citing:

```
@inproceedings{hui2024Mimic,
  title={Mimic: Speaking Style Disentanglement for Speech-Driven 3D Facial Animation},
  author={Hui Fu, Zeqing Wang, Ke Gong, Keze Wang, Tianshui Chen, Haojie Li, Haifeng Zeng, Wenxiong Kang},
  booktitle={The 38th Annual AAAI Conference on Artificial Intelligence (AAAI)},
  year={2024}
}
```
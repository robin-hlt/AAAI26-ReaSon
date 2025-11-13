# [AAAI 2026] ReaSon: Reinforced Causal Search with Information Bottleneck for Video Understanding

The official implementation of ReaSon


## 📢 News

- **[2025.11.13]** We realse codes of inference demo.
- **[2025.11.08]** 🎉🎉 Our paper **"ReaSon: Reinforced Causal Search with Information Bottleneck for Video Understanding"** has been **accepted to AAAI 2026**!

## 🧩 To-Do List

- [ ] 📄 Release the paper (arXiv preprint & project page)  
- [ ] 💻 Release full codes, including training and inference  
- [ ] 🚀 Release pretrained ReaSon policy  

## 🚀 Quick Start

<details>
<summary>🔧 Environment Setup</summary>

We provide a one-click installation script:

```bash
bash install.sh
```

Or install manually:

```bash
conda create -n reason python=3.9 -y
conda activate reason
git clone https://github.com/robin-hlt/AAAI26-ReaSon

# Install LLaVA-Video (optional)
git clone https://github.com/LLaVA-VL/LLaVA-NeXT
cd LLaVA-NeXT && pip install -e . && cd ..

# Install YOLO-World
git clone --recursive https://github.com/AILab-CVC/YOLO-World.git
cd YOLO-World && pip install -e . && cd ..

# Install ReaSon dependencies
pip install -r requirements_basic.txt
pip install "flash-attn==2.6.3" --no-build-isolation
```

</details>

---

<details>
<summary>🎬 Inference Demo</summary>

Run ReaSon to perform **reinforced causal search** and answer video questions:

```bash
python demo_reason.py \
    --video path/to/video.mp4 \
    --question "What is the person doing after opening the door?" \
    --save_dir outputs/demo
```

</details>


## 🙏 Acknowledgements

We sincerely thank the following open-source projects for providing essential components that contributed to our work

- [**TStar**](https://github.com/mll-lab-nu/TStar)
- [**LLaVA-Video**](https://github.com/LLaVA-VL/LLaVA-NeXT)

# [AAAI 2026] ReaSon: Reinforced Causal Search with Information Bottleneck for Video Understanding

The official implementation of ReaSon


## 📢 News

- **[2025.11.13]** We realse codes of inference demo.
- **[2025.11.08]** 🎉🎉 Our paper **"ReaSon: Reinforced Causal Search with Information Bottleneck for Video Understanding"** has been **accepted to AAAI 2026**!

## 🧩 To-Do List

- [ ] 📄 Release the paper (arXiv preprint & project page)  
- [ ] 💻 Release full codes, including training and inference  
- [ ] 🚀 Release pretrained ReaSon policy  

# 🔧 Installation

This section describes how to set up the environment for **ReaSon**.  
We recommend using **conda** to manage the Python environment.

---

## 1. Create Environment
```bash
conda create -n reason python=3.9 -y
conda activate reason
```

---

## 2. Install LLaVA-NeXT (optional)
LLaVA-NeXT is required only if you plan to train the selection policy or run local VLM inference.
```bash
git clone https://github.com/LLaVA-VL/LLaVA-NeXT
cd LLaVA-NeXT
pip install -e .
cd ..
```

---

## 3. Install YOLO-World (Object Detector)
```bash
git clone --recursive https://github.com/AILab-CVC/YOLO-World.git
cd YOLO-World

# Install PyTorch (CUDA 11.8)
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu118

# Install YOLO-World
pip install -e .
cd ..
```

---

## 4. Install ReaSon Dependencies
```bash
pip install -r requirements_basic.txt
pip install "flash-attn==2.6.3" --no-build-isolation
```

---

## 5. Fix mmdet / mmyolo Version Constraints
```bash
sed -i "s/mmcv_maximum_version = '2.1.0'/mmcv_maximum_version = '2.3.0'/g" \
$(python - << 'EOF'
import importlib.util
print(importlib.util.find_spec('mmdet').origin)
EOF
)

sed -i "s/mmcv_maximum_version = '2.1.0'/mmcv_maximum_version = '2.3.0'/g" \
$(python - << 'EOF'
import importlib.util
print(importlib.util.find_spec('mmyolo').origin)
EOF
)
```

---

## 6. Download Pretrained YOLO-World Weights
```bash
mkdir -p pretrained/YOLO-World

wget -O pretrained/YOLO-World/yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth \
https://huggingface.co/wondervictor/YOLO-World/resolve/main/yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth
```

---

## 7. Download LVIS Metadata
```bash
mkdir -p data/coco/lvis
wget -O data/coco/lvis/lvis_v1_minival_inserted_image_name.json \
https://huggingface.co/GLIPModel/GLIP/resolve/main/lvis_v1_minival_inserted_image_name.json

mkdir -p data/texts
wget -O data/texts/lvis_v1_class_texts.json \
https://github.com/AILab-CVC/YOLO-World/raw/refs/heads/master/data/texts/lvis_v1_class_texts.json
```

---

Your environment is ready.

## 🙏 Acknowledgements

We sincerely thank the following open-source projects for providing essential components that contributed to our work

- [**TStar**](https://github.com/mll-lab-nu/TStar)
- [**LLaVA-Video**](https://github.com/LLaVA-VL/LLaVA-NeXT)

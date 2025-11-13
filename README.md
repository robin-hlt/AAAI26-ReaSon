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
git clone https://github.com/robin-hlt/AAAI26-ReaSon.git

# Install LLaVA-Video (optional)
git clone https://github.com/LLaVA-VL/LLaVA-NeXT
cd LLaVA-NeXT && pip install -e . && cd ..

# Install YOLO-World
git clone --recursive https://github.com/AILab-CVC/YOLO-World.git
cd YOLO-World && pip install -e . && cd ..

# Install ReaSon dependencies
pip install -r requirements_basic.txt
pip install "flash-attn==2.6.3" --no-build-isolation

# Fix mmdet/mmyolo related issues
sed -i "s/mmcv_maximum_version = '2.1.0'/mmcv_maximum_version = '2.3.0'/g" $(python -c "import importlib.util; filename=importlib.util.find_spec('mmdet').origin;print(filename)")
sed -i "s/mmcv_maximum_version = '2.1.0'/mmcv_maximum_version = '2.3.0'/g" $(python -c "import importlib.util; filename=importlib.util.find_spec('mmyolo').origin;print(filename)")
# pip install --upgrade setuptools

# Download model
mkdir pretrained && cd pretrained
mkdir YOLO-World && cd YOLO-World
wget https://huggingface.co/wondervictor/YOLO-World/resolve/main/yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth && cd ../..

# Download data
mkdir -p data/coco/lvis
wget -O data/coco/lvis/lvis_v1_minival_inserted_image_name.json https://huggingface.co/GLIPModel/GLIP/resolve/main/lvis_v1_minival_inserted_image_name.json
mkdir -p data/texts
wget -O data/texts/lvis_v1_class_texts.json https://github.com/AILab-CVC/YOLO-World/raw/refs/heads/master/data/texts/lvis_v1_class_texts.json

# Fix YOLO-World small bug
sed -i "s/self.text_feats, None/self.text_feats, _/g" YOLO-World/yolo_world/models/detectors/yolo_world.py
```

</details>


<details>
<summary>🎬 Inference Demo</summary>

Run demo_reason.py to perform **reinforced causal search** and answer video questions:

```bash
python demo_reason.py \
   --ann ann_for_test.json \
   --video-id 0074f737-11cb-497d-8d07-77c3a8127391
```

</details>


## 🙏 Acknowledgements

We sincerely thank the following open-source projects for providing essential components that contributed to our work

- [**TStar**](https://github.com/mll-lab-nu/TStar)
- [**LLaVA-Video**](https://github.com/LLaVA-VL/LLaVA-NeXT)

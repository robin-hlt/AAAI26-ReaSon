conda create -n reason python=3.9
conda activate reason
git clone https://github.com/robin-hlt/AAAI26-ReaSon.git
cd AAAI26-ReaSon

# Do this if you want to re-train the selectiob policy or use LLaVA-Video for inference.
# If you want to use GPT for quick start, no need to download LLaVA. LLaVA is used by default.
git clone https://github.com/LLaVA-VL/LLaVA-NeXT
cd LLaVA-NeXT && pip install -e . && cd ..

# Prepare YOLO-World as object detector
git clone --recursive https://github.com/AILab-CVC/YOLO-World.git
cd YOLO-World
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
pip install -e . && cd ..

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

# Fix YOLO-World text_feats unpacking bug
sed -i "s/self.text_feats, None/self.text_feats, _/g" YOLO-World/yolo_world/models/detectors/yolo_world.py


# [AAAI 2026] ReaSon: Reinforced Causal Search with Information Bottleneck for Video Understanding

The official implementation of "[ReaSon: Reinforced Causal Search with Information Bottleneck for Video Understanding](https://arxiv.org/abs/2511.12530)" (AAAI 2026) in Pytorch.


## 📢 News

- **[2025.11.16]** We realsed the paper [![arXiv](https://img.shields.io/badge/arXiv-2511.12530-b31b1b.svg)](https://arxiv.org/abs/2511.12530).
- **[2025.11.13]** We realsed codes of inference demo.
- **[2025.11.08]** 🎉🎉 Our paper **"ReaSon: Reinforced Causal Search with Information Bottleneck for Video Understanding"** has been **accepted to AAAI 2026**!

## 🧩 To-Do List

- [x] 📄 Release the paper (arXiv preprint & project page)
- [x] 🚀 Release checkpoint of ReaSon policy  
- [ ] 💻 Release full codes, including training and inference  
 

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
cd AAAI26-ReaSon

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
<summary>📁 Project Structure</summary>
 
```
AAAI26-ReaSon/
├── LLaVA-NeXT/                     # LLaVA-Video (or Qwen if you used Qwen)
├── checkpoints/                    # Save checkpoints
├── ReaSon/                         # Core implementation of the ReaSon framework
│   ├── interface_grounding.py      # Video–language grounding (LLaVA-Video)
│   ├── interface_heuristic.py      # YOLO-World heuristic object extraction
│   ├── interface_searcher.py       # Detection for candidate pool
│   ├── policy_core.py              # Policy network and trainer
│   ├── ReaSonFramework.py          # Reinforced causal search pipeline
│   └── utilites.py                 # Hepler and shared utilities
├── YOLO-World/                     # YOLO-World detector repo
├── test_video/                     # Example videos for demo
├── ann_for_test.json               # Annotation JSON for inference demo
├── demo_reason.py                  # Inference demo script
├── train.py                        # ReaSon training script
├── install.sh                      # Environment setup
├── requirements_basic.txt          # Basic dependencies
└── README.md                       # Documentation
```
 
<details>

<details>
<summary>🤗 Policy Checkpoints</summary>

| Model             | Description                 | Link |
|-------------------|-----------------------------|------|
| **ReaSon-Policy** | selection policy checkpoint | 🤗 [Hugging Face](https://huggingface.co/RobinHua/ReaSon/tree/main) |

</details>


<details>
<summary>🎬 Inference Demo</summary>

Download the policy checkpoint and place it place into `checkpoints/`. Run demo_reason.py to perform **reinforced causal search** and answer video questions:

```bash
python demo_reason.py \
   --ann ann_for_test.json \
   --video-id 0074f737-11cb-497d-8d07-77c3a8127391
```

</details>


## 📦 Train on Your Own Dataset

<details>
<summary>📘 Dataset Format</summary>

To train ReaSon on your own data, prepare a JSON file where each element represents one video–question sample.

Each item requires the following keys:

- **video_id**: unique identifier  
- **video_path**: path to the video file  
- **question**: natural language question  
- **options**: multi-choice text (single line or multi-line)  
- **answer**: ground-truth answer label (A/B/C/…)  

**Example:**

```json
[
  {
    "video_id": "0074f737-11cb-497d-8d07-77c3a8127391",
    "video_path": "/path/to/videos/0074f737-11cb-497d-8d07-77c3a8127391.mp4",
    "question": "Taking into account all the actions performed by C, what can you deduce about the primary objective and focus within the video content?",
    "options": "A) C is cooking. B) C is doing laundry. C) C is cleaning the kitchen. D) C is cleaning dishes. E) C is cleaning the bathroom.",
    "answer": "D"
  },
  {
    "video_id": "00b9a0de-c59e-49cb-a127-6081e2fb8c8e",
    "video_path": "/path/to/videos/00b9a0de-c59e-49cb-a127-6081e2fb8c8e.mp4",
    "question": "What was the primary purpose of the cup of water in this video, and how did it contribute to the overall painting process?",
    "options": "A) To provide a source of water for the paintbrush. B) To provide a place to store the paintbrush. C) To provide a place to dispose of the paintbrush. D) To provide a place to rest the paintbrush. E) To clean the paintbrush.",
    "answer": "E"
  }
]
```

</details>

<details>
<summary>🛠️ Training Script</summary>

Run the following command to train ReaSon:

```bash
python train.py \
    --data-json your_dataset.json \
    --save-dir checkpoints/
```

</details>



## 🙏 Acknowledgements

We sincerely thank the following open-source projects for providing essential components that contributed to our work

- [**TStar**](https://github.com/mll-lab-nu/TStar)
- [**LLaVA-Video**](https://github.com/LLaVA-VL/LLaVA-NeXT)

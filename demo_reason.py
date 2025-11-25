import os
import gc
import json
import argparse
import torch
from ReaSon.utilites import BLIPEncoder
from ReaSon.policy_core import LSTMPolicyNet
from ReaSon.ReaSonFramework import ReaSonFramework, initialize_heuristic
from ReaSon.interface_grounding import ReaSonUniversalGrounder
import time

import warnings
warnings.filterwarnings("ignore")
# Path to selection policy checkpoint
CHECKPOINT_PATH = "./checkpoints/policy_weights.pth"


def load_record_by_id(ann_path, video_id):
    """
    Load a single record from the annotation JSON by matching video_id.
    Returns the record dict or None if not found.
    """
    try:
        records = json.load(open(ann_path, 'r', encoding='utf-8'))
    except Exception as e:
        print(f"[WARNING] Could not open annotation file '{ann_path}': {e}")
        return None

    for rec in records:
        if rec.get("video_id") == video_id:
            return rec
    print(f"[WARNING] No entry for video_id='{video_id}' in '{ann_path}'.")
    return None


def run_single(video_path, question, options, groundtruth=None):
    """
    Run ReaSonFramework on one video-question pair.
    Prints the predicted answer and, if provided, whether it is correct.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # initialize grounding, heuristic, BLIP encoder, and policy network
    grounder = ReaSonUniversalGrounder(
        model_name="llava",
        model_path="lmms-lab/LLaVA-Video-7B-Qwen2",
        gpt4_api_key="your gpt api key",
        num_frames=8
    )
    heuristic = initialize_heuristic(heuristic_type="yolo-World")
    blip = BLIPEncoder(
        model_name="Salesforce/blip-itm-base",
        device=device
    )

    policy_net = LSTMPolicyNet(input_dim=1536).to(device)
    policy_net.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    policy_net.eval()

    framework = ReaSonFramework(
        video_path=video_path,
        question=question,
        options=options,
        grounder=grounder,
        heuristic=heuristic,
        search_nframes=32,
        grid_rows=4,
        grid_cols=4,
        confidence_threshold=0.7,
        search_budget=0.5
    )
    start = time.time()
    answer, frames = framework.run_test(
        frame_encoder=lambda f: blip.encode_frames(f, text=question),
        text_encoder=blip.encode_text,
        policy_net=policy_net,
        traj_length=8
    )
    end = time.time()
    print(f"\nPredicted answer: {answer}")
    if groundtruth is not None:
        is_correct = (answer.strip().upper() == groundtruth.strip().upper())
        print(f"Ground-truth    : {groundtruth}  →  {'Correct' if is_correct else 'Incorrect'}")
    print(f"Time elapsed: {end - start} seconds")
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test a single video–question pair with ReaSonFramework"
    )
    parser.add_argument("-a", "--ann",
                        help="Path to annotation JSON (if provided, only --video-id is required)")
    parser.add_argument("-i", "--video-id",
                        help="Video ID to look up in annotation file")
    parser.add_argument("-v", "--video",
                        help="Path to the video file (required if no annotation file)")
    parser.add_argument("-q", "--question",
                        help="Question text (required if no annotation file)")
    parser.add_argument("-o", "--options",
                        help="Comma-separated list of answer options, e.g. \"A) Red,B) Blue,...\" (required if no annotation file)")
    parser.add_argument("-g", "--ground",
                        help="Optional ground-truth answer (overrides annotation)")

    args = parser.parse_args()

    # If annotation file is provided, only video_id is needed
    if args.ann:
        if not args.video_id:
            parser.error("When --ann is used, you must also specify --video-id")
        record = load_record_by_id(args.ann, args.video_id)
        if record is None:
            parser.error(f"No record found for video_id={args.video_id}")
        video_path = record["video_path"]
        question   = record["question"]
        ground_gt  = args.ground or record.get("answer")
        # parse options from annotation: split on newline
        opts_raw = record.get("options", "")
        options   = [opt.strip() for opt in opts_raw.split("\n") if opt.strip()]
    else:
        # no annotation: require video, question, options
        if not (args.video and args.question and args.options):
            parser.error("Without --ann, you must provide --video, --question, and --options")
        video_path = args.video
        question   = args.question
        ground_gt  = args.ground
        options    = [opt.strip() for opt in args.options.split(",") if opt.strip()]

    run_single(video_path, question, options, groundtruth=ground_gt)
    

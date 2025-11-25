import os
import gc
import json
import random
import argparse
import torch
import time

from ReaSon.interface_grounding import ReaSonUniversalGrounder
from ReaSon.ReaSonFramework import ReaSonFramework, initialize_heuristic
from ReaSon.policy_core import LSTMPolicyNet, TrajectorySampler, Trainer
from ReaSon.utilites import BLIPEncoder


def build_trainer(
    policy_net, optimizer, sampler,
    grounder, blip_encoder,
    question, options,
    gt_answer,
    target_objects, candidate_objects
):
    return Trainer(
        policy_net=policy_net,
        blip_encoder=blip_encoder,
        optimizer=optimizer,
        sampler=sampler,
        grounder=grounder,
        question=question,
        options=options,
        gt_answer=gt_answer,
        target_objects=target_objects,
        candidate_objects=candidate_objects
    )


def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    with open(args.data_json, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    total_steps = len(dataset)
    print(f"[INFO] Loaded dataset: {total_steps} samples")

    # Initialize components
    grounder = ReaSonUniversalGrounder(
        model_name="llava",
        model_path="lmms-lab/LLaVA-Video-7B-Qwen2",
        gpt4_api_key=None,
        num_frames=8
    )

    heuristic = initialize_heuristic(heuristic_type="yolo-world")

    blip_encoder = BLIPEncoder(
        model_name="Salesforce/blip-itm-base",
        device=device
    )

    policy_net = LSTMPolicyNet(input_dim=1536).to(device)

    # Resume (optional)
    if args.resume and os.path.exists(args.resume):
        print(f"[INFO] Loading checkpoint from {args.resume}")
        policy_net.load_state_dict(torch.load(args.resume, map_location=device))

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-4)
    sampler = TrajectorySampler(num_groups=4, traj_length=8)

    os.makedirs(args.save_dir, exist_ok=True)

    random.shuffle(dataset)

    for step, item in enumerate(dataset, start=1):
        print(f"[Step {step}/{total_steps}] video_id = {item.get('video_id','Unknown')}")

        try:
            question = item["question"]
            options = item.get("options", "")
            video_path = item["video_path"]
            gt_answer = item["answer"]

            reason_env = ReaSonFramework(
                video_path=video_path,
                question=question,
                options=options,
                grounder=grounder,
                heuristic=heuristic,
                search_nframes=32,
                grid_rows=4,
                grid_cols=4,
                confidence_threshold=0.7,
                search_budget=1.0
            )


            reason_env.run_train(
                frame_encoder=lambda frames: blip_encoder.encode_frames(frames, text=question),
                text_encoder=blip_encoder.encode_text,
                trainer_factory_fn=lambda grounder, question, options, target_objects, candidate_objects:
                    build_trainer(
                        policy_net=policy_net,
                        optimizer=optimizer,
                        sampler=sampler,
                        grounder=grounder,
                        blip_encoder=blip_encoder,
                        question=question,
                        options=options,
                        gt_answer=gt_answer,
                        target_objects=target_objects,
                        candidate_objects=candidate_objects
                    )
            )

        except Exception as e:
            print(f"[ERROR] Step {step} skipped: {e}")

        torch.cuda.empty_cache()
        gc.collect()

    # Save final policy
    save_path = os.path.join(args.save_dir, "policy_final.pth")
    torch.save(policy_net.state_dict(), save_path)
    print(f"[INFO] Saved final policy to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReaSon Training Script")

    parser.add_argument("--data-json", required=True, help="Path to dataset JSON")
    parser.add_argument("--save-dir", type=str, default="./checkpoints", help="Directory to save policy")
    parser.add_argument("--resume", type=str, default=None, help="Resume training from checkpoint")

    args = parser.parse_args()
    main(args)

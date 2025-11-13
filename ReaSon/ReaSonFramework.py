import os
import sys
import cv2
import logging
import numpy as np
import torch
from PIL import Image
from typing import List, Tuple
from ReaSon.interface_grounding import ReaSonUniversalGrounder
from ReaSon.interface_heuristic import YoloWorldInterface, OWLInterface, HeuristicInterface
from ReaSon.interface_searcher import ReaSonSearcher
from ReaSon.utilites import save_as_gif
import matplotlib.pyplot as plt
import pandas as pd
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class ReaSonFramework:
    """
    Main class for performing CIB-based keyframe selection and question-answering in a video.
    """

    def __init__(
        self,
        video_path: str,
        heuristic: HeuristicInterface,
        grounder: ReaSonUniversalGrounder,
        question: str,
        options: str,
        search_nframes: int = 8,
        grid_rows: int = 4,
        grid_cols: int = 4,
        output_dir: str = './output',
        confidence_threshold: float = 0.6,
        search_budget: int = 1000
    ):
        self.video_path = video_path
        self.grounder = grounder
        self.heuristic = heuristic
        self.question = question
        self.options = options
        self.search_nframes = search_nframes
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.output_dir = os.path.join(output_dir, os.path.basename(video_path).split('.')[0], question[:-1])
        self.confidence_threshold = confidence_threshold
        self.search_budget = search_budget
        self._create_output_dir()

        self.results = {} # to store search results, e.g., grounding, frames

    def _create_output_dir(self):
        """
        Ensure that the output directory exists.
        """
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self):
        """
        Run the ReaSon framework to search for objects and answer questions.
        """

        target_objects, cue_objects = self.get_grounded_objects()
        video_searcher = self.initialize_videoSearcher(target_objects, cue_objects)
        all_frames, time_stamps = self.perform_search(video_searcher, visualization=True)
        start_time = time.time()
        answer = self.perform_qa(all_frames)
        logger.info(f"Answer: {answer}")
        end_time = time.time()
        print(f"Total time: {end_time-start_time:.2f} seconds.")
        return {
            "Grounding Objects": {'target_objects': target_objects, 'cue_objects': cue_objects},
            "Frame Timestamps": time_stamps,
            "Answer": answer
        }

    def run_train(self, frame_encoder, text_encoder, trainer_factory_fn):
        """
        Train the selection policy via reinforcement learning.

        Args:
            frame_encoder: Callable, List[PIL.Image] → Tensor [16, D]
            text_encoder: Callable, str → Tensor [D]
            trainer_factory_fn: Callable, construct Trainer
        """
        # Detect visual elements（e.g. "cup", "table"）
        target_objects, cue_objects = self.get_grounded_objects()

        # construct candidate pool according to visual elements
        video_searcher = self.initialize_videoSearcher(target_objects, cue_objects)
        frames, time_stamps = self.perform_search(video_searcher)
        frames_pil = [Image.fromarray(frame.astype('uint8')) for frame in frames]

        # add candidate objects
        # candidate_objects = self.grounder.inference_grounding_candidate_boj(frames_pil,self.question)
        candidate_objects = ["placeholder"] # just a placeholder, useless

        # encode frames and question with BLIP
        frame_embs = frame_encoder(frames_pil)  # Tensor [16, D]
        question_emb = text_encoder(self.question)  # Tensor [D]

        torch.cuda.empty_cache()

        # 4. construct Trainer
        trainer = trainer_factory_fn(
            grounder=self.grounder,
            question=self.question,
            options=self.options,
            target_objects=target_objects,
            candidate_objects=candidate_objects
        )

        # forward once
        loss = trainer.train_step(frame_embs, question_emb, frames, time_stamps)

        print(f"[Train] RL loss = {loss:.4f}")


    def run_test(
            self,
            frame_encoder,
            text_encoder,
            policy_net,
            traj_length: int = 8
    ) -> Tuple[str, List[Image.Image]]:
        """
        Run the inference process and return the final answer
        """
        # === flag for visualization（only trial phase） ===
        do_visualize = False
        total_frames = 180

        # Detect visual elements
        target_objects, cue_objects = self.get_grounded_objects()

        # construct candidate pool
        searcher = self.initialize_videoSearcher(target_objects, cue_objects)
        frames, timestamps = self.perform_search(searcher,visualization=do_visualize)

        # encode frames and question
        frame_embs = frame_encoder(frames)  # [N, D]
        question_emb = text_encoder(self.question)  # [D]
        question_exp = question_emb.unsqueeze(0).repeat(frame_embs.size(0), 1)
        input_tensor = torch.cat([frame_embs, question_exp], dim=-1).unsqueeze(0)  # [1, N, 2D]

        # greedy frame selection
        policy_net.eval()
        with torch.no_grad():
            logits = policy_net(input_tensor)[0]  # [N]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            K = min(traj_length, probs.shape[0])
            topk = np.argsort(probs)[-K:]
            topk = np.sort(topk)

        # record the selected frame id
        if do_visualize:
            txt_path = os.path.join(self.output_dir, "frame_probs.txt")
            with open(txt_path, 'w') as f:
                f.write("timestamp\tprobability\n")
                for t, p in zip(timestamps, probs):
                    f.write(f"{t:.3f}\t{p:.6f}\n")
            print(f"Saved frame probabilities to {txt_path}")

            sel_txt = os.path.join(self.output_dir, "selected_frames.txt")
            with open(sel_txt, "w") as f:
                f.write("frame_idx\ttimestamp\n")
                for idx in topk:
                    f.write(f"{idx}\t{timestamps[int(idx)]:.3f}\n")
            print(f"Saved selected frame list to {sel_txt}")

            sel_dir = os.path.join(self.output_dir, "selected_frames")
            os.makedirs(sel_dir, exist_ok=True)
            for idx in topk:
                img = Image.fromarray(frames[int(idx)])
                img.save(os.path.join(sel_dir, f"frame_{int(idx):03d}.png"))
            print(f"Saved {len(topk)} selected frames to {sel_dir}")


        # generate answer
        selected_frames = [Image.fromarray(frames[i]) for i in topk]
        answer = self.grounder.inference_qa(selected_frames, self.question, self.options)

        # visualization
        if do_visualize:
            full_idx = np.arange(total_frames)
            series = pd.Series(data=np.nan,index=full_idx)
            series.loc[timestamps] = probs
            series_interp = series.interpolate(method="linear",limit_direction='both')
            smooth_probs = series_interp.values

            plt.figure(figsize=(10, 4))
            plt.plot(full_idx, smooth_probs, label="Smoothed Probability", alpha=0.3)
            #plt.scatter(timestamps, probs, s=20, alpha=0.6, label="Raw Probability")
            plt.xlabel("Timestamp (s)")
            plt.ylabel("Frame Selection Probability")
            plt.title("Policy Network Frame Probabilities")
            plt.legend(loc="best")
            plot_path = os.path.join(self.output_dir, "frame_probs_curve.svg")
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            print(f"Saved selected probability curve to {plot_path}")

        return answer.strip(), selected_frames

    def get_grounded_objects(self) -> Tuple[List[str], List[str]]:
        """
        Use Grounder to obtain target and cue objects.
        """
        target_objects, cue_objects = self.grounder.inference_query_grounding(
            video_path=self.video_path,
            question=self.question,
            options=self.options
        )
        self.results["Grounding Objects"] = {"target_objects": target_objects, "cue_objects":cue_objects}
        logger.info(f"Target objects: {target_objects}")
        logger.info(f"Cue objects: {cue_objects}")
        return target_objects, cue_objects


    def initialize_videoSearcher(self, target_objects: List[str], cue_objects: List[str]) -> ReaSonSearcher:
        """
        Initialize and configure the ReaSonSearcher with the given objects.
        """
        videoSearcher =  ReaSonSearcher(
            video_path=self.video_path,
            target_objects=target_objects,
            cue_objects=cue_objects,
            search_nframes=self.search_nframes,
            image_grid_shape=(self.grid_rows, self.grid_cols),
            output_dir=self.output_dir,
            confidence_threshold=self.confidence_threshold,
            search_budget=self.search_budget,
            heuristic=self.heuristic
        )

        return videoSearcher

    def perform_search(self, video_searcher: ReaSonSearcher, visualization: bool = False) -> Tuple[List[np.ndarray], List[float]]:
        """
        Perform the search for relevant frames and their timestamps.
        """
        if visualization:
            all_frames, time_stamps = video_searcher.search()
            self._save_frames(all_frames, time_stamps)
            self._save_searching_iterations(video_searcher)
            self._plot_and_save_scores(video_searcher)
            prob = video_searcher.score_distribution
            txt_path = "score_dis.txt"
            np.savetxt(txt_path,prob,fmt="%.6f")
        else:
            all_frames, time_stamps = video_searcher.search()
        
        logger.info(f"Found {len(all_frames)} frames, timestamps: {time_stamps}")
        return all_frames, time_stamps

    def perform_qa(self, frames: List[np.ndarray]) -> str:
        """
        Perform question answering on the retrieved frames.
        """
        return self.grounder.inference_qa(
            frames=frames,
            question=self.question,
            options=self.options
        )

    def _save_frames(self, frames: List[np.ndarray], timestamps: List[float]):
        """
        Save the relevant frames as image files.
        """
        frame_dir = os.path.join(self.output_dir, "frames")
        os.makedirs(frame_dir, exist_ok=True)

        for idx, (frame, timestamp) in enumerate(zip(frames, timestamps)):
            frame_path = os.path.join(frame_dir, f"frame_{idx}_at_{timestamp:.2f}s.jpg")
            cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            logger.info(f"Saved frame to {frame_path}")

    def _save_searching_iterations(self, video_searcher: ReaSonSearcher):
        """
        Save the frames and their annotations from search iterations.
        """
        image_grid_iters = video_searcher.image_grid_iters
        detect_annotot_iters = video_searcher.detect_annotot_iters
        
        for b in range(len(image_grid_iters[0])):
            images = [image_grid_iter[b] for image_grid_iter in image_grid_iters]
            anno_images = [detect_annotot_iter[b] for detect_annotot_iter in detect_annotot_iters]
            output_video_path = os.path.join(self.output_dir, f"search_iterations.gif")
            save_as_gif(images=anno_images, output_gif_path=output_video_path)
            logger.info(f"Saved search iterations GIF to {output_video_path}")

    def _plot_and_save_scores(self, video_searcher: ReaSonSearcher):
        """
        Plot and save the score distribution from the search process.
        """
        plot_path = os.path.join(self.output_dir, "score_distribution.png")
        video_searcher.plot_score_distribution(save_path=plot_path)
        logger.info(f"Score distribution plot saved to {plot_path}")


def initialize_heuristic(heuristic_type: str = "owl-vit") -> HeuristicInterface:
    """
    Initialize the object detection model based on the selected heuristic type.
    """
    if heuristic_type == 'owl-vit':
        model_name = "google/owlvit-base-patch32"
        owl_interface = OWLInterface(model_name_or_path=model_name)
        logger.info("OWLInterface initialized successfully.")
        return owl_interface
    elif heuristic_type == 'yolo-World':
        config_path = "./YOLO-World/configs/pretrain/yolo_world_v2_xl_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py"
        checkpoint_path = "./pretrained/YOLO-World/yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth"
        yolo_interface = YoloWorldInterface(config_path=config_path, checkpoint_path=checkpoint_path)
        logger.info("YoloWorldInterface initialized successfully.")
        return yolo_interface
    else:
        raise NotImplementedError(f"Heuristic type '{heuristic_type}' is not implemented.")


def run_reason(
    video_path: str,
    question: str,
    options: str,
    grounder: str = "llava",
    heuristic: str = "yolo-World",
    search_nframes: int = 8,
    grid_rows: int = 4,
    grid_cols: int = 4,
    confidence_threshold: float = 0.6,
    search_budget: float = 0.5,
    output_dir: str = './output'
):
    """
    Execute the ReaSon video frame search and question-answering process.
    """
    grounder = ReaSonUniversalGrounder(model_name=grounder,
                                      gpt4_api_key="your gpt api key",
                                      model_path="lmms-lab/LLaVA-Video-7B-Qwen2")
    heuristic = initialize_heuristic(heuristic)

    ReaSonQA = ReaSonFramework(
        video_path=video_path,
        grounder=grounder,
        heuristic=heuristic,
        question=question,
        options=options,
        search_nframes=search_nframes,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        output_dir=output_dir,
        confidence_threshold=confidence_threshold,
        search_budget=search_budget
    )

    return ReaSonQA.run()

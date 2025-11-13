import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from PIL import Image, ImageDraw
from ReaSon.utilites import parase_options
import random
import gc


class LSTMPolicyNet(nn.Module):
    """
    A 2-layer LSTM policy network for frame selection.
    Input: [B, 16, 2D] or [16, 2D]
    Output: [B, 16] or [16]
    """
    def __init__(self, input_dim: int, hidden_dim: int = 512, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )
        self.output_head = nn.Linear(hidden_dim, 1)

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        if input_seq.dim() == 2:
            input_seq = input_seq.unsqueeze(0)
            single_input = True
        else:
            single_input = False
        lstm_out, _ = self.lstm(input_seq)
        logits = self.output_head(lstm_out).squeeze(-1)
        return logits.squeeze(0) if single_input else logits


class TrajectorySampler:
    def __init__(self, num_groups: int = 4, traj_length: int = 4, max_attempts: int = 2):
        self.M = num_groups
        self.K = traj_length
        self.max_attempts = max_attempts

    def sample(self, probs: torch.Tensor) -> List[List[int]]:
        """
        Sample M diverse trajectories of length K using softmax probabilities.
        Ensures diversity by deduplicating sorted trajectories.
        """
        sampled_set = set()
        attempts = 0
        N=probs.shape[0]
        K=min(self.K,N)

        while len(sampled_set) < self.M and attempts < self.M * self.max_attempts:
            try:
                traj = torch.multinomial(probs, K, replacement=False)
                traj_sorted = tuple(sorted(traj.tolist()))
                sampled_set.add(traj_sorted)
            except Exception as e:
                print(f"[Sampler] Warning: multinomial sampling failed: {e}")
            attempts += 1

        # If there are no enough samples, complement by random sampling
        while len(sampled_set) < self.M:
            traj = random.sample(range(len(probs)), K)
            sampled_set.add(tuple(sorted(traj)))

        return [list(traj) for traj in sampled_set]


class GRPOTrainer:
    def __init__(
        self,
        policy_net,
        optimizer,
        sampler,
        grounder,
        blip_encoder,
        question: str,
        options: str,
        gt_answer: str,
        target_objects: List[str],
        candidate_objects: List[str],
        lambda_cycle: float = 0.5
    ):
        self.policy_net = policy_net
        self.optimizer = optimizer
        self.sampler = sampler
        self.grounder = grounder
        self.blip_encoder = blip_encoder
        self.question = question
        self.options = options
        self.gt_answer = gt_answer
        self.target_objects = set([t.lower() for t in target_objects])
        self.candidate_objects = set([c.lower() for c in candidate_objects])
        self.lambda_cycle = lambda_cycle

    def train_step(self,
                   frame_embs: torch.Tensor,
                   question_emb: torch.Tensor,
                   frame_pool: List[Image.Image],
                   timestamps: List[float]) -> float:
        question_exp = question_emb.unsqueeze(0).repeat(frame_embs.size(0), 1)
        input_tensor = torch.cat([frame_embs, question_exp], dim=-1).unsqueeze(0)

        policy_logits = self.policy_net(input_tensor)
        probs = F.softmax(policy_logits.squeeze(0), dim=-1)
        sampled_groups = self.sampler.sample(probs)

        # construct counterfactual prob and sample a counterfactual trajectory
        cf_probs_raw = 1.0 - probs
        cf_probs = cf_probs_raw / cf_probs_raw.sum()

        try:
            cf_traj = torch.multinomial(cf_probs, min(self.sampler.K, len(cf_probs)), replacement=False).tolist()
            cf_traj = sorted(cf_traj)
        except Exception as e:
            print(f"[CF Sampler] Warning: multinomial sampling failed: {e}")
            cf_traj = random.sample(range(len(cf_probs)), min(self.sampler.K, len(cf_probs)))
            cf_traj = sorted(cf_traj)

        traj_frames_cf, _ = self._get_frames_by_indices(frame_pool, frame_embs, timestamps, cf_traj)
        try:
            vlm_logits_cf = self.grounder.inference_logits(traj_frames_cf, self.question, self.options)
        except Exception as e:
            print(f"[Causal ERROR] {e}")
            vlm_logits_cf = None

        group_rewards = []
        cycle_rewards = []
        answer_rewards = []
        causal_rewards = []

        for i, traj in enumerate(sampled_groups):
            traj_frames, traj_ts = self._get_frames_by_indices(frame_pool, frame_embs, timestamps, traj)
            try:
                vlm_logits = self.grounder.inference_logits(traj_frames, self.question, self.options)
                answer = self.grounder.inference_qa(traj_frames, self.question, self.options)
            except Exception as e:
                print(f"[Ground ERROR] {e}")
                answer = "ERROR"
                vlm_logits = None

            if vlm_logits is not None and vlm_logits_cf is not None:
                casual_reward = self.compute_causal_reward(vlm_logits, vlm_logits_cf)
            else:
                casual_reward = 0.0

            total_reward, cycle_reward, answer_reward = self._evaluate_reward(answer, traj_frames, casual_reward)
            group_rewards.append(total_reward)
            cycle_rewards.append(cycle_reward)
            answer_rewards.append(answer_reward)
            causal_rewards.append(casual_reward)
            torch.cuda.empty_cache()
            gc.collect()

        loss = self._compute_grpo_loss(sampled_groups, probs, group_rewards)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train_epoch(self, dataset: List[Tuple[torch.Tensor, torch.Tensor]]):
        total_loss = 0.0
        for i, (frame_embs, question_emb) in enumerate(dataset):
            loss = self.train_step(frame_embs, question_emb)
            print(f"[Train] Sample {i + 1}/{len(dataset)} - Loss: {loss:.4f}")
            total_loss += loss

        avg_loss = total_loss / len(dataset)
        print(f"[Train] Epoch finished. Average Loss: {avg_loss:.4f}")
        return avg_loss

    @staticmethod
    def _get_frames_by_indices(frame_pool: List[Image.Image],
                               frame_embs: torch.tensor,
                               timestamps: List[float],
                               indices: List[int]):
        images = [frame_pool[i] for i in indices]
        ts = [timestamps[i] for i in indices]
        #embs = frame_embs[indices]
        return images,ts

    def compute_semantic_iou(self,pred_list,target_list,threshold=0.7):
        match_count=0
        pred_set=set(pred_list)
        target_set=set(target_list)
        if not pred_set or not target_set:
            return 0.0
        target_embs = [self.blip_encoder.encode_text(t) for t in target_set]

        for p in pred_set:
            p_emb = self.blip_encoder.encode_text(p)
            sims = [F.cosine_similarity(p_emb,t_emb, dim=0) for t_emb in target_embs]
            if max(sims,default=0) >= threshold:
                match_count += 1
        union_count = len(pred_set.union(target_set))
        return match_count/union_count if union_count > 0 else 0.0

    def compute_causal_reward(
            self,
            vlm_logits: torch.Tensor,
            vlm_logits_cf: torch.Tensor,
            reduction: str = "batchmean"
    ) -> float:
        """
        Compute the causal reward based on the KL divergence between
        original logits and counterfactual logits (after softmax).

        Args:
            vlm_logits: [1, T, V] tensor from original frames
            vlm_logits_cf: [1, T, V] tensor from counterfactual frames
            reduction: 'batchmean', 'mean' or 'sum' for KL

        Returns:
            Causal reward as a float
        """
        try:
            # 默认取最后一个 token（即最后输出位置）进行对比
            P = torch.softmax(vlm_logits[:, -1, :], dim=-1)
            P_prime = torch.softmax(vlm_logits_cf[:, -1, :], dim=-1)

            # KL(P || P')，注意：P.log() 是第一个参数
            kl = F.kl_div(P.log(), P_prime, reduction=reduction)
            return kl.item()
        except Exception as e:
            print(f"[KL ERROR] {e}")
            return 0.0

    def _evaluate_reward(
            self,
            predicted_answer: str,
            frames: List[Image.Image],
            causal_reward: float = 0.0
    ) -> Tuple[float, float, float]:
        """
        Compute total reward with answer/cycle/causal components.

        Returns:
            total_reward, cycle_reward, answer_reward, causal_reward
        """
        answer_reward = 1.0 if predicted_answer.strip().upper() == self.gt_answer.strip().upper() else 0.0

        try:
            options_dict = parase_options(self.options)
            answer_text = options_dict.get(predicted_answer)
            pred_obj_str = self.grounder.inference_cycle_qa(frames, self.question, answer_text)
            pred_objs = [x.strip().lower() for x in pred_obj_str.split(",") if x.strip()]
            cycle_reward = self.compute_semantic_iou(pred_objs, self.target_objects)
        except Exception as e:
            print(f"[REWARD ERROR] {e}")
            cycle_reward = 0.0

        total_reward = answer_reward + 0.5*cycle_reward + 0.5*causal_reward

        return total_reward, cycle_reward, answer_reward

    def _compute_log_prob(self, traj: List[int], probs: torch.Tensor) -> torch.Tensor:
        selected_probs = probs[traj] + 1e-8
        log_probs = torch.log(selected_probs)
        return log_probs.sum()

    def _compute_grpo_loss(self, sampled_groups, probs, group_rewards) -> torch.Tensor:
        group_rewards = torch.tensor(group_rewards, dtype=torch.float32, device=probs.device)
        loss = torch.tensor(0.0, dtype=torch.float32, device=probs.device)
        M = len(sampled_groups)
        baseline = group_rewards.mean()
        # std = group_rewards.std(unbiased=False) + 1e-6

        for traj, r in zip(sampled_groups, group_rewards):
            log_pi = self._compute_log_prob(traj, probs)
            # advantage = (r - baseline) / std
            advantage = r - baseline
            loss += -advantage * log_pi

        policy_loss = loss / M
        return policy_loss

    def save_model(self, path: str):
        torch.save(self.policy_net.state_dict(), path)
        print(f"[GRPOTrainer] Policy model saved to {path}")

    def load_model(self, path: str):
        state_dict = torch.load(path, map_location='cpu')
        self.policy_net.load_state_dict(state_dict)
        print(f"[GRPOTrainer] Policy model loaded from {path}")

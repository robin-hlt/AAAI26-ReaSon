import os
import random
from typing import Dict, Optional, List
import openai
from openai import OpenAI
from PIL import Image
import re
import numpy as np
import torch
from ReaSon.utilites import encode_image_to_base64, load_video_frames
from decord import VideoReader, cpu
import copy
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates


class LlavaInterface:
    """
    Inference interface for lmms-lab/LLaVA-Video-7B-Qwen2.
    Supports both video file input (via video_path) and raw image frames (List[PIL.Image]).
    Unified with QwenInterface-style: main entry is inference_with_frames().
    """

    def __init__(
        self,
        model_path: str = "lmms-lab/LLaVA-Video-7B-Qwen2",
        model_base: Optional[str] = None,
        device: str = "cuda",
        torch_dtype="bfloat16",
    ):
        """
        Initialize LLaVA video model and processor.
        """
        self.device = device
        self.tokenizer, self.model, self.processor, _ = load_pretrained_model(
            model_path,
            model_base,
            "llava_qwen",
            torch_dtype=torch_dtype,
            device_map="auto",
        )
        self.model.eval()
        print(f"[LlavaInterface] Model loaded from {model_path} on {device}")

    def load_video_frames(
        self,
        video_path: str,
        max_frames: int = 64,
        fps: int = 1,
        force_sample: bool = True
    ) -> tuple[np.ndarray, str, float]:
        """
        Load video and extract uniformly sampled frames using Decord.

        Returns:
            frames: np.ndarray of shape (N, H, W, C)
            frame_time_str: comma-separated string of frame times
            video_duration: total length of the video in seconds
        """
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        total_frames = len(vr)
        raw_fps = vr.get_avg_fps()
        stride = max(1, round(raw_fps / fps))
        frame_idx = list(range(0, total_frames, stride))
        frame_time = [i / raw_fps for i in frame_idx]

        if len(frame_idx) > max_frames or force_sample:
            frame_idx = np.linspace(0, total_frames - 1, max_frames, dtype=int).tolist()
            frame_time = [i / raw_fps for i in frame_idx]

        frames = vr.get_batch(frame_idx).asnumpy()
        frame_time_str = ",".join([f"{t:.2f}s" for t in frame_time])
        video_duration = total_frames / raw_fps
        return frames, frame_time_str, video_duration

    def inference_with_frames(
        self,
        query: str,
        frames: Optional[List[Image.Image]] = None,
        video_path: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        max_frames: int = 64,
        fps: int = 1,
    ) -> str:
        """
        Unified inference interface. Supports either video_path or preloaded frames.

        Args:
            query: Prompt string.
            frames: Optional list of PIL.Image frames.
            video_path: Optional path to video file.
            temperature: Generation temperature.
            max_tokens: Max new tokens to generate.
            max_frames: Max frames to use if extracting from video.
            fps: Sampling rate if extracting from video.

        Returns:
            Model-generated answer string.
        """
        assert video_path is not None or frames is not None, "Either video_path or frames must be provided."

        if video_path:
            np_frames, frame_time_str, video_duration = self.load_video_frames(
                video_path, max_frames=max_frames, fps=fps
            )
            processed = self.processor.preprocess(np_frames, return_tensors="pt")["pixel_values"]
            frame_info = (
                f"The video lasts for {video_duration:.2f} seconds, "
                f"and {processed.shape[0]} frames are uniformly sampled. "
                f"These frames are located at {frame_time_str}."
            )
        else:
            processed = self.processor.preprocess(frames, return_tensors="pt")["pixel_values"]
            frame_info = (
                f"{processed.shape[0]} frames are provided. "
                f"Please analyze the visual content across them."
            )

        video_tensor = [img.unsqueeze(0).to(device=self.device,dtype=torch.bfloat16) for img in processed]

        # Prompt construction
        conv = copy.deepcopy(conv_templates["qwen_2"])
        full_prompt = frame_info + "\n" + query.strip()
        conv.append_message(conv.roles[0], full_prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Tokenize
        input_ids = tokenizer_image_token(
            prompt,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt"
        ).unsqueeze(0).to(self.device)

        # Generate
        output_ids = self.model.generate(
            inputs=input_ids,
            images=video_tensor,
            modalities=["video"],
            temperature=temperature,
            max_new_tokens=max_tokens,
            do_sample=False
        )
        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    def get_logits(
            self,
            query: str,
            frames: Optional[List[Image.Image]] = None,
            video_path: Optional[str] = None,
            temperature: float = 0.7,
            max_tokens: int = 1024,
            max_frames: int = 64,
            fps: int = 1,
    ) -> dict:
        """
        Given a prompt and list of frames, returns:
        - logits: the raw model logits
        - probs: softmax probability over vocab for last token
        """
        assert video_path is not None or frames is not None, "Either video_path or frames must be provided."

        if video_path:
            np_frames, frame_time_str, video_duration = self.load_video_frames(
                video_path, max_frames=max_frames, fps=fps
            )
            processed = self.processor.preprocess(np_frames, return_tensors="pt")["pixel_values"]
            frame_info = (
                f"The video lasts for {video_duration:.2f} seconds, "
                f"and {processed.shape[0]} frames are uniformly sampled. "
                f"These frames are located at {frame_time_str}."
            )
        else:
            processed = self.processor.preprocess(frames, return_tensors="pt")["pixel_values"]
            frame_info = (
                f"{processed.shape[0]} frames are provided. "
                f"Please analyze the visual content across them."
            )

        video_tensor = [img.unsqueeze(0).to(device=self.device,dtype=torch.bfloat16) for img in processed]

        # Prompt construction
        conv = copy.deepcopy(conv_templates["qwen_2"])
        full_prompt = frame_info + "\n" + query.strip()
        conv.append_message(conv.roles[0], full_prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Tokenize
        input_ids = tokenizer_image_token(
            prompt,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt"
        ).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            # 前向推理 logits
            outputs = self.model(
                input_ids=input_ids,
                images=video_tensor,
                modalities=["video"],
                return_dict=True
            )
            logits = outputs.logits

        return logits

    def inference(
        self,
        query: str,
        frames: Optional[List[Image.Image]] = None,
        video_path: Optional[str] = None,
        max_new_tokens: int = 1024,
    ) -> str:
        """
        Legacy-compatible interface.
        Delegates to inference_with_frames().
        """
        return self.inference_with_frames(
            query=query,
            frames=frames,
            video_path=video_path,
            max_tokens=max_new_tokens
        )

from typing import List, Optional
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
# from qwen_vl_utils import process_vision_info

class QwenInterface:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "cuda"
    ):
        """
        Initialize Qwen model and processor.
        """

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        self.device = self.model.device
        self.processor = AutoProcessor.from_pretrained(model_name)

    def inference_with_frames(
        self,
        query: str,
        frames: Optional[List[Image.Image]] = None,
        temperature: float = 0.7,
        max_tokens: int = 128
    ) -> str:
        """
        Unified inference interface that supports mixed text and image inputs.
        The query may include "<image>" tags to indicate where to insert images.

        Args:
            query: The query text which can include "<image>" tags.
            frames: A list of PIL Image objects corresponding to the <image> tags.
            temperature: Sampling temperature for generation.
            max_tokens: Maximum number of new tokens to generate.

        Returns:
            The generated answer as a trimmed string.
        """
        # Build the message list by splitting the query at <image> tags.
        messages = []
        content_list = []
        parts = query.split("<image>")
        for i, part in enumerate(parts):
            if part.strip():
                content_list.append({"type": "text", "text": part.strip()})
            if frames and i < len(frames):
                # Directly pass the PIL Image; the processor will handle it.
                content_list.append({"type": "image", "image": frames[i]})
        if not content_list:
            content_list.append({"type": "text", "text": query})
        messages.append({"role": "user", "content": content_list})

        # Generate a text template using the processor's chat template function.
        # This step integrates both text and vision context.
        text_template = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Optionally, process vision inputs if your setup requires it.
        # image_inputs, video_inputs = process_vision_info(messages)

        # Prepare the inputs for the model.
        inputs = self.processor(
            text=[text_template],
            images=frames,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.device)

        # Generate the output.
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=False
        )
        # Trim the input tokens so that only the generated answer remains.
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0].strip()

    def inference(
        self,
        query: str,
        frames: Optional[List[Image.Image]] = None,
        max_new_tokens: int = 128
    ) -> str:
        """
        Legacy inference method. For compatibility, it calls inference_with_frames.
        """
        if frames is None:
            frames = []
        return self.inference_with_frames(
            query=query,
            frames=frames,
            max_tokens=max_new_tokens
        )

class GPT4Interface:
    def __init__(self, model: str = "gpt-4o", api_key: Optional[str] = None):
        """
        Initialize the GPT-4 API client. The API key is read from the environment
        variable OPENAI_API_KEY if not provided.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model
        if not self.api_key:
            raise ValueError("Environment variable OPENAI_API_KEY is not set.")
        # openai.api_key = self.api_key
        self.clinet = OpenAI(base_url="https://xiaoai.plus/v1",
                api_key=api_key)

    def _build_messages(self, system_message: str, user_content: List) -> List[Dict]:
        """
        Build the messages list required by the OpenAI API.
        """
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content},
        ]

    def _encode_frames(self, frames: List[Image.Image]) -> List[Dict]:
        """
        Encode image frames into Base64 formatted messages.
        """
        messages = []
        for i, frame in enumerate(frames):
            try:
                frame_base64 = encode_image_to_base64(frame)
                visual_context = {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{frame_base64}",
                        "detail": "low"
                    }
                }
                messages.append(visual_context)
            except Exception as e:
                raise ValueError(f"Error encoding frame {i}: {str(e)}")
        return messages

    def inference_text_only(
        self,
        query: str,
        system_message: str = "You are a helpful assistant.",
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """
        Perform inference using GPT-4 API for text-only input.
        """
        messages = self._build_messages(system_message, query)
        try:
            response = self.clinet.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error: {str(e)}"

    def _inference_with_frames(
        self,
        query: str,
        frames: List[Image.Image],
        system_message: str = "You are a helpful assistant.",
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """
        Perform inference using GPT-4 API with frames as context.
        """
        user_content = [{"type": "text", "text": query}]
        try:
            user_content.extend(self._encode_frames(frames))
        except ValueError as e:
            return str(e)
        messages = self._build_messages(system_message, user_content)
        try:
            response = self.clinet.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error: {str(e)}"

    def inference_qa(
        self,
        question: str,
        options: str,
        frames: Optional[List[Image.Image]] = None,
        system_message: str = "You are a helpful assistant.",
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> str:
        """
        Perform multiple-choice inference using GPT-4 API.
        
        Args:
            question: The question to answer.
            options: Multiple-choice options as a string.
            frames: Optional visual context.
        
        Returns:
            The selected option (e.g., A, B, C, D).
        """
        query = (
            f"Question: {question}\nOptions: {options}\n"
            "Answer with the letter corresponding to the best choice."
        )
        user_content = [{"type": "text", "text": query}]
        if frames:
            try:
                user_content.extend(self._encode_frames(frames))
            except ValueError as e:
                return str(e)
        messages = self._build_messages(system_message, user_content)
        try:
            response = self.clinet.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error: {str(e)}"

    def inference_with_frames(
        self,
        query: str,
        frames: List[Image.Image],
        system_message: str = "You are a helpful assistant.",
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """
        A unified inference interface supporting mixed text and image inputs.
        The query may include <image> tags.
        """
        parts = query.split("<image>")
        user_content = []
        for i, part in enumerate(parts):
            if part.strip():
                user_content.append({"type": "text", "text": part.strip()})
            if i < len(frames):
                try:
                    frame_base64 = encode_image_to_base64(frames[i])
                    visual_context = {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{frame_base64}",
                            "detail": "low"
                        }
                    }
                    user_content.append(visual_context)
                except Exception as e:
                    return f"Error encoding frame {i}: {str(e)}"
        messages = self._build_messages(system_message, user_content)
        try:
            response = self.clinet.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error: {str(e)}"


class ReaSonUniversalGrounder:
    """
    Combines functionalities of ReaSonGrounder and ReaSonGPTGrounder.
    Allows switching between LlavaInterface and GPT4Interface via the backend parameter.
    """
    def __init__(
        self,
        model_name: str = "gpt-4o",
        model_path: Optional[str] = None,
        model_base: Optional[str] = None,
        gpt4_api_key: Optional[str] = None,
        num_frames: Optional[int] = 8,
    ):
        self.backend = model_name.lower()
        self.num_frames = num_frames
        if "llava" in self.backend:
            if not model_path:
                raise ValueError("Please provide model_path for LlavaInterface")
            self.VLM_model_interface = LlavaInterface(model_path=model_path, model_base=model_base)
        elif "qwen" in self.backend:
            # Initialize QwenInterface if 'qwen' is specified in the backend.
            self.VLM_model_interface = QwenInterface(model_name=model_name,device="auto")
        elif "gpt" in self.backend:
            self.VLM_model_interface = GPT4Interface(model=model_name, api_key=gpt4_api_key)
        else:
            raise ValueError("backend must be one of: 'llava', 'qwen', or 'gpt4'.")

    def inference_query_grounding(
        self,
        video_path: str,
        question: str,
        options: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 512
    ) -> Dict[str, List[str]]:
        """
        Identify target objects and cue objects from the video based on the question.
        
        Args:
            video_path: Path to the video file.
            question: The question.
            options: (Optional) multiple-choice options.
        
        Returns:
            A dictionary with two keys: target_objects and cue_objects.
        """
        frames = load_video_frames(video_path=video_path, num_frames=self.num_frames)
        system_prompt = (
            "You are an assistant helping to predict specific objects about question." +
            f"\nHere is a question about the video: {question}\n" +
            "Here is a video:\n" + "\n".join(["<image>"] * len(frames))
        )
        system_prompt += (
            "\nConsider both question text and video frames:\n"
            "1. Identify key objects that can locate the answer (list key objects, separated by commas).\n"
            "2. Identify cue objects that might be near the key objects and appear in the scenes (list cue objects, separated by commas).\n\n"
            "Listing the key objects and cue objects in two lines separated by commas."
        )
        response = self.VLM_model_interface.inference_with_frames(
            query=system_prompt,
            frames=frames,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        lines = [line.strip() for line in response.split("\n") if line.strip()]
        if len(lines) == 2:

            target_objects = [self.check_objects_str(obj) for obj in lines[0].split(",") if obj.strip()]
            cue_objects = [self.check_objects_str(obj) for obj in lines[1].split(",") if obj.strip()]
        elif len(lines) == 1:
            t_objects = lines[0].split(";")[0]
            try:
                c_objects = lines[0].split(";")[1]
            except:
                c_objects = ''
            target_objects = [self.check_objects_str(obj) for obj in t_objects.split(",") if obj.strip()]
            cue_objects = [self.check_objects_str(obj) for obj in c_objects.split(",") if obj.strip()]
        else:
            raise ValueError(f"Unexpected response format --> {response}")

        return target_objects, cue_objects

    def inference_grounding_candidate_boj(
            self,
            frames: List[Image.Image],
            question: str,
            temperature: float = 0.0,
            max_tokens: int = 512,
            max_frames: int = 8
    ) -> List[str]:
        if len(frames) > max_frames:
            frames = random.sample(frames,max_frames)
        system_prompt = (
                "Here is a video:\n" + "\n".join(["<image>"] * len(frames)) +
                "\nHere is a question about the video:\n" +
                f"Question: {question}\n"
                "\nWhen answering this question about the video:\n"
                "Identify key objects that help locate the answer (list them, separated by commas).\n"
                "Do not include explanations or cue objects. Just list the key objects."
        )

        response = self.VLM_model_interface.inference_with_frames(
            query=system_prompt,
            frames=frames,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        lines = [line.strip() for line in response.split("\n") if line.strip()]
        objects_line = lines[0] if lines else ""
        target_objects = [self.check_objects_str(obj) for obj in objects_line.split(",") if obj.strip()]

        return target_objects

    def check_objects_str(self, obj: str) -> str:
        """
        Process the object string to normalize object names by:
        - Lowercasing
        - Removing prefixes like "1. ", "2. ", "Key objects:"
        - Removing punctuation
        - Stripping extra whitespace
        """
        obj = obj.strip().lower()

        # Remove known prefixes (with optional whitespace)
        obj = re.sub(r"^(key objects|cue objects)?[:\-]?\s*", "", obj)
        obj = obj.replace("key objects: ", "").replace("cue objects: ", "").replace(": ", "")
        obj = re.sub(r"^[0-9]+\.\s*", "", obj)  # e.g., "1. "
        
        # Remove punctuation like periods, colons etc.
        obj = re.sub(r"[^\w\s-]", "", obj)  # Keep letters, numbers, space, hyphen

        return obj.strip()

    def inference_qa(
        self,
        frames: List[Image.Image],
        question: str,
        options: str,
        temperature: float = 0.7,
        video_path: Optional[str] = None,
        max_tokens: int = 128
    ) -> str:
        """
        Perform multiple-choice inference and return the most likely option (e.g., A, B, C, D).
        """
        assert video_path is not None or frames is not None, "Either video_path or frames must be provided."

        if frames is None:
            frames = load_video_frames(video_path,num_frames=self.num_frames)

        system_prompt = (
            "The following images are sampled from a video in chronological order.\n" +
            "Please consider the temporal progression and cause-effect relations among objects and events.\n" +
            "Select the best answer to the multiple-choice question based on this information.\n" +
            "\n".join(["<image>"] * len(frames)) +
            f"\nQuestion: {question}\n" +
            f"Options: {options}\n\n" +
            "You must choose one and answer with the option's letter from the given choices directly."
        )
        response = self.VLM_model_interface.inference_with_frames(
            query=system_prompt,
            frames=frames,
            temperature=temperature,
            max_tokens=30
        )
        return response.strip()

    def inference_openend_qa(
        self,
        frames: List[Image.Image],
        question: str,
        temperature: float = 0.2,
        max_tokens: int = 2048
    ) -> str:
        """
        Perform open-ended question answering based on the video.
        """
        system_prompt = (
            "Answer the following question briefly based on the video.\n" +
            "\n".join(["<image>"] * len(frames)) +
            f"\nQuestion: {question}\n"
        )
        response = self.VLM_model_interface.inference_with_frames(
            query=system_prompt,
            frames=frames,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.strip()

    def inference_cycle_qa(
        self,
        frames: List[Image.Image],
        question: str,
        predicted_answer: str,
        temperature: float = 0.7,
        max_tokens: int = 512
    ) -> str:
        cycle_prompt = (
            f"The question is: {question}"
            f"The answer is: {predicted_answer}"
            "Based on both, what objects must appear in the video to support this answer?"
            "Directly list all such objects separated by commas. Do not add any explanation."
        )
        response = self.VLM_model_interface.inference_with_frames(
            query=cycle_prompt,
            frames=frames,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.strip()

    def inference_logits(
            self,
            frames: List[Image.Image],
            question: str,
            options: Optional[str] = None,
            temperature: float = 0.0,
            max_tokens: int = 128,
    ) -> dict:
        """
        Entry point to get both logits and answer from VLM.
        Delegates to LlavaInterface. Only supports 'llava' backend.
        """
        if "llava" not in self.backend:
            raise NotImplementedError("Only 'llava' backend supports logits extraction.")

        prompt = (
                "Select the best answer to the following multiple-choice question based on the video.\n" +
                "\n".join(["<image>"] * len(frames)) +
                f"\nQuestion: {question}\n"
        )
        if options:
            prompt += f"Options: {options}\n\n"
        prompt += "Answer with the option's letter from the given choices directly."

        return self.VLM_model_interface.get_logits(
            query=prompt,
            frames=frames,
            temperature=temperature,
            max_tokens=max_tokens
        )
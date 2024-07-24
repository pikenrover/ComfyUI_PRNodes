import os
import folder_paths
import random
import torch
import hashlib

import comfy.utils
import comfy.model_management

DELIMITER = '|\|'

RESOLUTIONS = [
    { 'width': 576, 'height': 576 },
    { 'width': 576, 'height': 1024 },
    { 'width': 1024, 'height': 576 },
    { 'width': 512, 'height': 512 },
    { 'width': 512, 'height': 768 },
    { 'width': 768, 'height': 512 },
]

MAX_RESOLUTION=16384

def addnet_hash_safetensors(b):
    """kohya-ss hash for safetensors from https://github.com/kohya-ss/sd-scripts/blob/main/library/train_util.py"""
    hash_sha256 = hashlib.sha256()
    blksize = 1024 * 1024

    b.seek(0)
    header = b.read(8)
    n = int.from_bytes(header, "little")

    offset = n + 8
    b.seek(offset)
    for chunk in iter(lambda: b.read(blksize), b""):
        hash_sha256.update(chunk)

    return str(hash_sha256.hexdigest())[0:12].lower()

def calculate_sha256(file_path):
    sha256_hash = hashlib.sha256()

    with open(file_path, "rb") as f:
        # Read the file in chunks to avoid loading the entire file into memory
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)

    return str(sha256_hash.hexdigest())[0:12].lower()

class RandomPrompt:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f.endswith(".prompt")]
        return {"required": {
            "prompts": [sorted(files), ], 
            "clip": ("CLIP", ),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),}}
    RETURN_TYPES = ("CONDITIONING","CONDITIONING","LATENT","WIDTH","HEIGHT","INT","STRING","STRING")
    FUNCTION = "encode"

    CATEGORY = "conditioning"

    def encode(self, prompts, clip, seed):
        input_dir = folder_paths.get_input_directory()
        prompt_filename = os.path.join(input_dir, prompts)
        prompt_list = []
        with open(prompt_filename) as file:
            while line := file.readline():
                if len(line.rstrip()) > 0:
                    prompt_list.append(line.rstrip())

        random.seed(seed)
        random_prompt = random.choice(prompt_list)
        random_prompt_split = random_prompt.split(DELIMITER)
        positive_prompt = random_prompt_split[0]
        negative_prompt = random_prompt_split[1] + ', child, loli, underage'

        positive_tokens = clip.tokenize(positive_prompt)
        positive_cond, positive_pooled = clip.encode_from_tokens(positive_tokens, return_pooled=True)

        negative_tokens = clip.tokenize(negative_prompt)
        negative_cond, negative_pooled = clip.encode_from_tokens(negative_tokens, return_pooled=True)

        resolution = random.choice(RESOLUTIONS)
        width = resolution['width']
        height = resolution['height']

        latent = torch.zeros([1, 4, height // 8, width // 8], device=self.device)

        return ([[positive_cond, {"pooled_output": positive_pooled, }]],
                [[negative_cond, {"pooled_output": negative_pooled, }]],
                {"samples":latent},
                width * 2,
                height * 2,
                seed,
                positive_prompt,
                negative_prompt)
    
class RandomPromptMixed:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f.endswith(".prompt")]
        return {"required": {
            "prompts": [sorted(files), ], 
            "clip": ("CLIP", ),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),}}
    RETURN_TYPES = ("CONDITIONING","CONDITIONING","LATENT","WIDTH","HEIGHT","INT","STRING","STRING")
    FUNCTION = "encode"

    CATEGORY = "conditioning"

    def encode(self, prompts, clip, seed):
        input_dir = folder_paths.get_input_directory()
        prompt_filename = os.path.join(input_dir, prompts)
        prompt_list = []
        with open(prompt_filename) as file:
            while line := file.readline():
                if len(line.rstrip()) > 0:
                    prompt_list.append(line.rstrip())

        bag_of_words = []
        prompt_lengths = []
        for prompt in prompt_list:
            prompt_split = prompt.split(DELIMITER)
            positive_prompt = prompt_split[0]
            prompt_length = 0
            prompt_split = positive_prompt.split(',')
            for word in prompt_split:
                bag_of_words.append(word)
                prompt_length += 1
            prompt_lengths.append(prompt_length)

        random.seed(seed)
        random_prompt = random.choice(prompt_list)
        random_prompt_split = random_prompt.split(DELIMITER)
        negative_prompt = random_prompt_split[1] + ', child, loli, underage'

        random_length = random.choice(prompt_lengths)
        index = 0
        prompt = []
        while index < random_length:
            prompt.append(random.choice(bag_of_words))
            index += 1
        positive_prompt = ','.join(prompt)

        print(positive_prompt)
        print(negative_prompt)

        positive_tokens = clip.tokenize(positive_prompt)
        positive_cond, positive_pooled = clip.encode_from_tokens(positive_tokens, return_pooled=True)

        negative_tokens = clip.tokenize(negative_prompt)
        negative_cond, negative_pooled = clip.encode_from_tokens(negative_tokens, return_pooled=True)

        resolution = random.choice(RESOLUTIONS)
        width = resolution['width']
        height = resolution['height']

        latent = torch.zeros([1, 4, height // 8, width // 8], device=self.device)

        return ([[positive_cond, {"pooled_output": positive_pooled, }]],
                [[negative_cond, {"pooled_output": negative_pooled, }]],
                {"samples":latent},
                width * 2,
                height * 2,
                seed,
                positive_prompt,
                negative_prompt)
    
class ImageScaleTo:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    crop_methods = ["disabled", "center"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image": ("IMAGE",), "upscale_method": (s.upscale_methods,),
                              "width": ("WIDTH",),
                              "height": ("HEIGHT",),
                              "crop": (s.crop_methods,)}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"

    CATEGORY = "image/upscaling"

    def upscale(self, image, upscale_method, width, height, crop):
        if width == 0 and height == 0:
            s = image
        else:
            samples = image.movedim(-1,1)

            if width == 0:
                width = max(1, round(samples.shape[3] * height / samples.shape[2]))
            elif height == 0:
                height = max(1, round(samples.shape[2] * width / samples.shape[3]))

            s = comfy.utils.common_upscale(samples, width, height, upscale_method, crop)
            s = s.movedim(1,-1)
        return (s,)
    
class EmptyLatentImageScaleBy:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "width": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                              "height": ("INT", {"default": 768, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                              "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                              "scale_by": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.1}),}}
    RETURN_TYPES = ("LATENT","WIDTH","HEIGHT")
    FUNCTION = "generate"

    CATEGORY = "latent"

    def generate(self, width, height, batch_size=1, scale_by=2):
        latent = torch.zeros([batch_size, 4, height // 8, width // 8], device=self.device)
        scaled_width = int(width * scale_by)
        scaled_height = int(height * scale_by)
        return ({"samples":latent}, scaled_width, scaled_height)

class LoraLoaderExtended:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": { 
                    "model": ("MODEL",),
                    "clip": ("CLIP", ),
                    "lora_name": ( folder_paths.get_filename_list("loras"), ),
                    "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                    "strength_clip": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                    },
                "optional": {
                    "lora_list": ("STRING", {"default": '', "multiline": True}),
                    "lora_hash_list": ("STRING", {"default": '', "multiline": True})
                }}
    RETURN_TYPES = ("MODEL", "CLIP", "STRING", "STRING")
    FUNCTION = "load_lora"

    CATEGORY = "loaders"

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip, lora_list='', lora_hash_list=''):
        if strength_model == 0 and strength_clip == 0:
            return (model, clip, lora_list, lora_hash_list)

        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                temp = self.loaded_lora
                self.loaded_lora = None
                del temp

        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

        model_lora, clip_lora = comfy.sd.load_lora_for_models(model,clip, lora, strength_model, strength_clip)

        target_len = lora_name.rfind('.')
        is_safetensors = str(lora_name).lower().find('.safetensors') >= 0
        lora_name_no_ext = lora_name[0:target_len]
        # ex <lora:princess_xl_v2:1> <lora:Expressive_H:1>
        if(len(lora_list) > 0):
            lora_list += ' '
        lora_list += '<lora:'
        lora_list += lora_name_no_ext
        lora_list += ':'
        lora_list += str(strength_model)
        lora_list += '>'

        # ex: princess_xl_v2: 12a7bd822b5f, Expressive_H: 5671f20a9a6b
        if(len(lora_hash_list) > 0):
            lora_hash_list += ', '
        lora_hash_list += lora_name_no_ext
        lora_hash_list += ': '
        if is_safetensors:
            with open(lora_path, "rb") as file:
                lora_hash_list += addnet_hash_safetensors(file)
        else:
            lora_hash_list += calculate_sha256(lora_path)

        return (model_lora, clip_lora, lora_list, lora_hash_list)

NODE_CLASS_MAPPINGS = {
    "RandomPrompt": RandomPrompt,
    "RandomPromptMixed": RandomPromptMixed,
    "ImageScaleTo": ImageScaleTo,
    "EmptyLatentImageScaleBy": EmptyLatentImageScaleBy,
    "LoraLoaderExtended": LoraLoaderExtended
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RandomPrompt": "RandomPrompt",
    "RandomPromptMixed": "RandomPromptMixed",
    "ImageScaleTo": "ImageScaleTo",
    "EmptyLatentImageScaleBy": "EmptyLatentImageScaleBy",
    "LoraLoaderExtended": "LoraLoaderExtended",
}

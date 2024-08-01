import os
import folder_paths
import random
import torch
import hashlib

from datetime import datetime
import json
import piexif
import piexif.helper
from PIL import Image, ImageOps, ImageSequence, ExifTags, ImageFile, UnidentifiedImageError
from PIL.PngImagePlugin import PngInfo
import numpy as np
import folder_paths
import comfy.sd
from nodes import MAX_RESOLUTION

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

addnet_hash_cache = {}
def addnet_hash_safetensors(file_path):
    if file_path in addnet_hash_cache:
        return addnet_hash_cache[file_path]

    """kohya-ss hash for safetensors from https://github.com/kohya-ss/sd-scripts/blob/main/library/train_util.py"""
    with open(file_path, "rb") as b:
        hash_sha256 = hashlib.sha256()
        blksize = 1024 * 1024

        b.seek(0)
        header = b.read(8)
        n = int.from_bytes(header, "little")

        offset = n + 8
        b.seek(offset)
        for chunk in iter(lambda: b.read(blksize), b""):
            hash_sha256.update(chunk)

        hash = str(hash_sha256.hexdigest())[0:12].lower()
        addnet_hash_cache[file_path] = hash
        return hash

sha256_hash_cache = {}
def calculate_sha256(file_path):
    if file_path in sha256_hash_cache:
        return sha256_hash_cache[file_path]

    sha256_hash = hashlib.sha256()

    with open(file_path, "rb") as f:
        # Read the file in chunks to avoid loading the entire file into memory
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)

    hash = str(sha256_hash.hexdigest())[0:12].lower()
    return hash

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
    RETURN_TYPES = ("CONDITIONING","CONDITIONING","LATENT","WIDTH","HEIGHT","STRING","STRING")
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
        positive_prompt = random_prompt_split[0] + ', embedding:safe_pos'
        negative_prompt = random_prompt_split[1] + ', child, loli, underage, embedding:SimpleNegativeV3, embedding:safe_neg,'

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
    RETURN_TYPES = ("CONDITIONING","CONDITIONING","LATENT","WIDTH","HEIGHT","STRING","STRING")
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
        negative_prompt = random_prompt_split[1] + ', child, loli, underage, embedding:SimpleNegativeV3, embedding:safe_neg'

        random_length = random.choice(prompt_lengths)
        index = 0
        prompt = []
        while index < random_length:
            prompt.append(random.choice(bag_of_words))
            index += 1
        positive_prompt = ','.join(prompt) + ', embedding:safe_pos'

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

        print(model)

        model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)

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
            lora_hash_list += addnet_hash_safetensors(lora_path)
        else:
            lora_hash_list += calculate_sha256(lora_path)

        return (model_lora, clip_lora, lora_list, lora_hash_list)
    
def handle_whitespace(string: str):
    return string.strip().replace("\n", " ").replace("\r", " ").replace("\t", " ")


def get_timestamp(time_format):
    now = datetime.now()
    try:
        timestamp = now.strftime(time_format)
    except:
        timestamp = now.strftime("%Y-%m-%d-%H%M%S")

    return timestamp


def make_pathname(filename, seed, modelname, counter, time_format):
    filename = filename.replace("%date", get_timestamp("%Y-%m-%d"))
    filename = filename.replace("%time", get_timestamp(time_format))
    filename = filename.replace("%model", modelname)
    filename = filename.replace("%seed", str(seed))
    filename = filename.replace("%counter", str(counter))
    return filename


def make_filename(filename, seed, modelname, counter, time_format):
    filename = make_pathname(filename, seed, modelname, counter, time_format)

    return get_timestamp(time_format) if filename == "" else filename

def parse_name(ckpt_name):
    path = ckpt_name
    filename = path.split("/")[-1]
    filename = filename.split(".")[:-1]
    filename = ".".join(filename)
    return filename

class ImageSaveWithMetadata:
    def __init__(self):
        self.output_dir = folder_paths.output_directory

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", ),
                "filename": ("STRING", {"default": f'%time_%seed', "multiline": False}),
                "path": ("STRING", {"default": '', "multiline": False}),
                "extension": (['png', 'jpeg', 'webp'],),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "ckpt_name": ("CKPT_NAME", {"default": '', "multiline": False}),
                "ckpt_hash": ("CKPT_HASH", {"default": '', "multiline": False}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "positive": ("STRING", {"default": 'unknown', "multiline": True}),
                "negative": ("STRING", {"default": 'unknown', "multiline": True}),
                "width": ("WIDTH", {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 8}),
                "height": ("HEIGHT", {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 8}),
                "lossless_webp": ("BOOLEAN", {"default": True}),
                "quality_jpeg_or_webp": ("INT", {"default": 100, "min": 1, "max": 100}),
                "counter": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff }),
                "time_format": ("STRING", {"default": "%Y-%m-%d-%H%M%S", "multiline": False}),
                "lora_list": ("STRING", {"default": '', "multiline": True}),
                "lora_hash_list": ("STRING", {"default": '', "multiline": True})
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_files"

    OUTPUT_NODE = True

    CATEGORY = "ImageSaverTools"

    def save_files(self, images, width, height, positive, negative, steps, cfg, sampler_name, scheduler, seed, ckpt_name, ckpt_hash, quality_jpeg_or_webp,
                   lossless_webp, counter, filename, path, extension, time_format, lora_list, lora_hash_list, prompt=None, extra_pnginfo=None):
        filename = make_filename(filename, seed, ckpt_name, counter, time_format,)
        path = make_pathname(path, seed, ckpt_name, counter, time_format)
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        basemodelname = parse_name(ckpt_name)
        modelhash = ckpt_hash
        positive = positive + ', ' + lora_list
        sampler_name = str(sampler_name).replace('euler_ancestral', 'Euler a')
        comment = f"{handle_whitespace(positive)}\nNegative prompt: {handle_whitespace(negative)}\nSteps: \
{steps}, Sampler: {sampler_name}{f'_{scheduler}' if scheduler != 'normal' else ''}, CFG Scale: {cfg}, \
Seed: {seed}, Size: {width}x{height}, Model hash: {modelhash}, Model: {basemodelname}, \
Lora hashes: \"{lora_hash_list}\", Version: ComfyUI"
        output_path = os.path.join(self.output_dir, path)

        if output_path.strip() != '':
            if not os.path.exists(output_path.strip()):
                print(f'The path `{output_path.strip()}` specified doesn\'t exist! Creating directory.')
                os.makedirs(output_path, exist_ok=True)    

        filenames = self.save_images(images, output_path, filename, comment, extension, quality_jpeg_or_webp, lossless_webp, prompt, extra_pnginfo)

        subfolder = os.path.normpath(path)
        return {"ui": {"images": map(lambda filename: {"filename": filename, "subfolder": subfolder if subfolder != '.' else '', "type": 'output'}, filenames)}}

    def save_images(self, images, output_path, filename_prefix, comment, extension, quality_jpeg_or_webp, lossless_webp, prompt=None, extra_pnginfo=None) -> list[str]:
        img_count = 1
        paths = list()
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            if images.size()[0] > 1:
                filename_prefix += "_{:02d}".format(img_count)

            if extension == 'png':
                metadata = PngInfo()
                metadata.add_text("parameters", comment)

                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

                filename = f"{filename_prefix}.png"
                img.save(os.path.join(output_path, filename), pnginfo=metadata, optimize=True)
            else:
                filename = f"{filename_prefix}.{extension}"
                file = os.path.join(output_path, filename)
                img.save(file, optimize=True, quality=quality_jpeg_or_webp, lossless=lossless_webp)
                exif_bytes = piexif.dump({
                    "Exif": {
                        piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(comment, encoding="unicode")
                    },
                })
                piexif.insert(exif_bytes, file)

            paths.append(filename)
            img_count += 1
        return paths
    
class CheckpointLoaderSimpleExtended:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                             }}
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "CKPT_NAME", "CKPT_HASH")
    FUNCTION = "load_checkpoint"

    CATEGORY = "loaders"

    def load_checkpoint(self, ckpt_name):
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(ckpt_path,
                                                    output_vae=True,
                                                    output_clip=True,
                                                    embedding_directory=folder_paths.get_folder_paths("embeddings"))[:3]

        print('hashing model...')
        ckpt_hash = calculate_sha256(ckpt_path)[:10]
        print('done')
        
        return (out[0], out[1], out[2], ckpt_name, ckpt_hash)

def pillow(fn, arg):
    prev_value = None
    try:
        x = fn(arg)
    except (OSError, UnidentifiedImageError, ValueError): #PIL issues #4472 and #2445, also fixes ComfyUI issue #3416
        prev_value = ImageFile.LOAD_TRUNCATED_IMAGES
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        x = fn(arg)
    finally:
        if prev_value is not None:
            ImageFile.LOAD_TRUNCATED_IMAGES = prev_value
        return x
    
def strip_path(path):
    #This leaves whitespace inside quotes and only a single "
    #thus ' ""test"' -> '"test'
    #consider path.strip(string.whitespace+"\"")
    #or weightier re.fullmatch("[\\s\"]*(.+?)[\\s\"]*", path).group(1)
    path = path.strip()
    if path.startswith("\""):
        path = path[1:]
    if path.endswith("\""):
        path = path[:-1]
    return path
    
def get_sorted_dir_files_from_directory(directory: str, skip_first_images: int=0, select_every_nth: int=1, extensions=None):
    directory = strip_path(directory)
    dir_files = os.listdir(directory)
    dir_files = sorted(dir_files)
    dir_files = [os.path.join(directory, x) for x in dir_files]
    dir_files = list(filter(lambda filepath: os.path.isfile(filepath), dir_files))
    # filter by extension, if needed
    if extensions is not None:
        extensions = list(extensions)
        new_dir_files = []
        for filepath in dir_files:
            ext = "." + filepath.split(".")[-1]
            if ext.lower() in extensions:
                new_dir_files.append(filepath)
        dir_files = new_dir_files
    # start at skip_first_images
    dir_files = dir_files[skip_first_images:]
    dir_files = dir_files[0::select_every_nth]
    return dir_files
    
class LoadRandomImage:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        directories = []
        for item in os.listdir(input_dir):
            if not os.path.isfile(os.path.join(input_dir, item)) and item != "clipspace":
                directories.append(item)
        return {"required": { 
            "directory": (directories,),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    CATEGORY = "image"

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"
    def load_image(self, directory, seed):
        input_dir = folder_paths.get_input_directory() + "\\" + directory
        images = get_sorted_dir_files_from_directory(input_dir)
        random.seed(seed)
        image_path = random.choice(images)
        
        img = pillow(Image.open, image_path)
        
        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ['MPO']
        
        for i in ImageSequence.Iterator(img):
            i = pillow(ImageOps.exif_transpose, i)

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]
            
            if image.size[0] != w or image.size[1] != h:
                continue
            
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask)

    @classmethod
    def IS_CHANGED(s, directory: str, **kwargs):
        if directory is None:
            return ""
        
        return directory

    @classmethod
    def VALIDATE_INPUTS(s, directory: str):
        if not folder_paths.exists_annotated_filepath(directory):
            return "Invalid folder: {}".format(directory)

        return True

NODE_CLASS_MAPPINGS = {
    "RandomPrompt": RandomPrompt,
    "RandomPromptMixed": RandomPromptMixed,
    "ImageScaleTo": ImageScaleTo,
    "EmptyLatentImageScaleBy": EmptyLatentImageScaleBy,
    "LoraLoaderExtended": LoraLoaderExtended,
    "Save Image w/Metadata": ImageSaveWithMetadata,
    "CheckpointLoaderSimpleExtended": CheckpointLoaderSimpleExtended,
    "LoadRandomImage": LoadRandomImage
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RandomPrompt": "RandomPrompt",
    "RandomPromptMixed": "RandomPromptMixed",
    "ImageScaleTo": "ImageScaleTo",
    "EmptyLatentImageScaleBy": "EmptyLatentImageScaleBy",
    "LoraLoaderExtended": "LoraLoaderExtended",
    "Save Image w/Metadata": "Save Image w/Metadata",
    "CheckpointLoaderSimpleExtended": "CheckpointLoaderSimpleExtended",
    "LoadRandomImage": "LoadRandomImage"
}

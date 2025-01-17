import os
from types import MethodType
from pathlib import Path

import torch
import torchvision.transforms as transforms
import cv2
from annotator.hed import SOFT_HEDdetector
from annotator.lineart import LineartDetector
from diffusers import UNet2DConditionModel, ControlNetModel
from transformers import CLIPVisionModelWithProjection, AutoProcessor, LlavaNextForConditionalGeneration
from PIL import Image
from huggingface_hub import snapshot_download
from ip_adapter import StyleShot, StyleContentStableDiffusionControlNetPipeline
import argparse

import logging
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def preproc_img(image):
    x, y = image.size
    h = w = 512
    image = transforms.CenterCrop(min(x,y))(image)
    image = image.resize((w, h), resample=Image.Resampling.LANCZOS)
    return image

def main(args):
    torch.cuda.set_device(f'cuda:{args.device}')

    base_model_path = "runwayml/stable-diffusion-v1-5"
    transformer_block_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    
    device = "cuda"

    if args.preprocessor == "Lineart":
        detector = LineartDetector()
        styleshot_model_path = "StyleShot_lineart"
    elif args.preprocessor == "Contour":
        detector = SOFT_HEDdetector()
        styleshot_model_path = "StyleShot"
    else:
        raise ValueError("Invalid preprocessor")

    if not os.path.isdir(styleshot_model_path):
        styleshot_model_path = snapshot_download(styleshot_model_path, local_dir=styleshot_model_path)
        print(f"Downloaded model to {styleshot_model_path}")

    # weights for ip-adapter and our content-fusion encoder
    if not os.path.isdir(base_model_path):
        base_model_path = snapshot_download(base_model_path, local_dir=base_model_path)
        print(f"Downloaded model to {base_model_path}")
    if not os.path.isdir(transformer_block_path):
        transformer_block_path = snapshot_download(transformer_block_path, local_dir=transformer_block_path)
        print(f"Downloaded model to {transformer_block_path}")

    ip_ckpt = os.path.join(styleshot_model_path, "pretrained_weight/ip.bin")
    style_aware_encoder_path = os.path.join(styleshot_model_path, "pretrained_weight/style_aware_encoder.bin")

    unet = UNet2DConditionModel.from_pretrained(base_model_path, subfolder="unet")
    content_fusion_encoder = ControlNetModel.from_unet(unet)
    
    pipe = StyleContentStableDiffusionControlNetPipeline.from_pretrained(base_model_path, controlnet=content_fusion_encoder)
    styleshot = StyleShot(device, pipe, ip_ckpt, style_aware_encoder_path, transformer_block_path)

    style_img_list = sorted(os.listdir(args.style))
    content_img_list = sorted(os.listdir(args.content))

    # prepare vlm
    if args.extract_prompt:
        i2p_model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
        i2p_processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
        prompt = "[INST] <image>\nWhat is portrayed on the image, describe shortly. [/INST]"

    for sty_img_path in style_img_list:
        for cnt_img_path in content_img_list:
            style_image = Image.open(os.path.join(args.style, sty_img_path))
            # processing content image
            content_image = cv2.imread(os.path.join(args.content, cnt_img_path))
            content_image = cv2.cvtColor(content_image, cv2.COLOR_BGR2RGB)
            content_image = detector(content_image)
            content_image = Image.fromarray(content_image)

            style_image = preproc_img(style_image)
            content_image = preproc_img(content_image)

            # extract prompt
            if args.extract_prompt:
                i2p_inputs = i2p_processor(images=content_image, text=prompt, return_tensors="pt")
                generate_ids = i2p_model.generate(**i2p_inputs, max_length=8096)
                content_img_descr = i2p_processor.batch_decode(
                    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                content_img_descr = content_img_descr.split('[/INST]')[-1].lstrip()
            else:
                content_img_descr = args.prompt

            log.info(f'Content image description: {content_img_descr}')

            generation = styleshot.generate(
                style_image=style_image, prompt=[[content_img_descr]], content_image=content_image
            )

            sty_name = Path(sty_img_path).stem
            cnt_name = Path(cnt_img_path).stem
            generation[0][0].save(os.path.join(args.output, f'{sty_name}_{cnt_name}.png'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--style", type=str, default="style.png")
    parser.add_argument("--content", type=str, default="content.png")
    parser.add_argument("--preprocessor", type=str, default="Contour", choices=["Contour", "Lineart"])
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--output", type=str, default="output")
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--extract_prompt", action='store_true')
    args = parser.parse_args()
    main(args)
    print('--- SUCCESS ---')

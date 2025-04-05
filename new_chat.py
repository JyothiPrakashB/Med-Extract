#!/usr/bin/env python3

"""
Multimodal Medical Image + Hinglish Text Inference Script
This script loads a fine-tuned Zephyr 7B model with ViT image encoder and generates
English summaries from medical images and Hinglish text inputs.

Usage:
    python zephyr_inference.py --checkpoint /path/to/checkpoint.pt --image /path/to/image.jpg --text "Your Hinglish text here"

Requirements:
    pip install torch transformers pillow argparse tqdm peft
"""

import os
import argparse
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig, 
    AutoImageProcessor, 
    ViTModel
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

class MultimodalLLM(nn.Module):
    def __init__(self, model_name):
        super(MultimodalLLM, self).__init__()
        print("Initializing ViT image encoder...")
        self.image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        self.vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

        for name, param in self.vit_model.named_parameters():
            param.requires_grad = False

        print("Initializing projection layer...")
        self.projection = nn.Linear(768, 4096)

        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loading LLM model: {model_name}...")
        
        try:
            # Try to load with BitsAndBytes if available
            self.quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                quantization_config=self.quant_config
            )
            print("Loaded model with 4-bit quantization")
            
            for name, param in self.llm_model.named_parameters():
                param.requires_grad = False
                
            self.llm_model.config.use_cache = False
            self.kbit_model = prepare_model_for_kbit_training(self.llm_model)
            
            self.config = LoraConfig(
                r=8, 
                lora_alpha=32, 
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                    "lm_head",
                ],
                lora_dropout=0.05, 
                bias="none", 
                task_type="CAUSAL_LM"
            )
            
            self.adapter_model = get_peft_model(self.kbit_model, self.config)
            
        except Exception as e:
            print(f"Could not load with BitsAndBytes, falling back to fp16: {e}")
            # Fallback to fp16 if BitsAndBytes is not available
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            self.llm_model.config.use_cache = False
            self.config = LoraConfig(
                r=8, 
                lora_alpha=32, 
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                    "lm_head",
                ],
                lora_dropout=0.05, 
                bias="none", 
                task_type="CAUSAL_LM"
            )
            self.adapter_model = get_peft_model(self.llm_model, self.config)

        self.max_length = 512
        print("Model initialization complete")

    def load_state_dict(self, state_dict, strict=False):
        """
        Custom state dict loading to handle mismatches between the checkpoint and model structure.
        """
        print("Using custom state_dict loading to handle key mismatches...")
        
        # Create a new state dict with remapped keys
        new_state_dict = {}
        
        # First copy all keys that exactly match
        model_keys = dict(self.named_parameters())
        for key in state_dict:
            if key in model_keys:
                new_state_dict[key] = state_dict[key]
        
        # Handle special cases for lm_head weights
        if "llm_model.lm_head.weight" in state_dict and "llm_model.lm_head.base_layer.weight" in model_keys:
            new_state_dict["llm_model.lm_head.base_layer.weight"] = state_dict["llm_model.lm_head.weight"]
        
        if "kbit_model.lm_head.weight" in state_dict and "kbit_model.lm_head.base_layer.weight" in model_keys:
            new_state_dict["kbit_model.lm_head.base_layer.weight"] = state_dict["kbit_model.lm_head.weight"]
            
        if "adapter_model.base_model.model.lm_head.weight" in state_dict and "adapter_model.base_model.model.lm_head.base_layer.weight" in model_keys:
            new_state_dict["adapter_model.base_model.model.lm_head.base_layer.weight"] = state_dict["adapter_model.base_model.model.lm_head.weight"]
        
        # For LoRA keys, check if we need to update the patterns
        lora_keys = {}
        for key in state_dict:
            if "lora_A" in key or "lora_B" in key:
                lora_keys[key] = state_dict[key]
        
        # Now load the state dict with the updated keys
        print(f"Remapped {len(new_state_dict)} keys from the original state dict")
        return super().load_state_dict(new_state_dict, strict=strict)

    def visual_encoder(self, image):
        """Process image through ViT and projection layer"""
        inputs = self.image_processor(image, return_tensors="pt")
        device = next(self.parameters()).device
        inputs.pixel_values = inputs.pixel_values.to(device)

        with torch.no_grad():
            outputs = self.vit_model(pixel_values=inputs.pixel_values)

        last_hidden_states = outputs.last_hidden_state
        img_proj = self.projection(last_hidden_states)
        att_img = torch.ones([img_proj.shape[0], img_proj.shape[1]+1],
                           dtype=inputs.pixel_values.dtype,
                           device=device) 

        return att_img, img_proj

    def forward(self, text, image):
        """Forward pass for training/validation"""
        self.tokenizer.padding_side = "right"
        att_img, img_proj = self.visual_encoder(image)

        tokenized_text = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding="longest", 
            truncation=True, 
            max_length=self.max_length,  
            add_special_tokens=False
        )
        
        device = next(self.parameters()).device
        tokenized_text.input_ids = tokenized_text.input_ids.to(device)
        tokenized_text.attention_mask = tokenized_text.attention_mask.to(device)
        
        # Get the right embed_tokens function from the model
        if hasattr(self, 'adapter_model') and hasattr(self.adapter_model.model, 'model'):
            embed_fn = self.adapter_model.model.model.embed_tokens
        else:
            embed_fn = self.llm_model.model.embed_tokens

        text_embeds = embed_fn(tokenized_text.input_ids)

        bos = torch.ones([tokenized_text.input_ids.shape[0], 1],
                         dtype=tokenized_text.input_ids.dtype,
                         device=device) * self.tokenizer.bos_token_id

        bos_embeds = embed_fn(bos)
        input_embeds = torch.cat([bos_embeds, img_proj, text_embeds], dim=1)
        attention_mask = torch.cat([att_img, tokenized_text.attention_mask], dim=1)

        targets = tokenized_text.input_ids.masked_fill(
            tokenized_text.input_ids == self.tokenizer.pad_token_id, -100
        )

        empty_targets = (
            torch.ones([img_proj.shape[0], img_proj.shape[1]+1], dtype=torch.long).to(device).fill_(-100)  # plus one for bos
        )

        targets = torch.cat([empty_targets, targets], dim=1)

        if hasattr(self, 'adapter_model'):
            outputs = self.adapter_model(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        else:
            outputs = self.llm_model(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )

        return outputs

    def generate(self, text, image, max_length=128):
        """Generate text based on image and text inputs"""
        self.tokenizer.padding_side = "right"
        att_img, img_proj = self.visual_encoder(image)

        tokenized_text = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding="longest", 
            truncation=True, 
            max_length=self.max_length,  
            add_special_tokens=False
        )
        
        device = next(self.parameters()).device
        tokenized_text.input_ids = tokenized_text.input_ids.to(device)
        tokenized_text.attention_mask = tokenized_text.attention_mask.to(device)
        
        # Get the right embed_tokens function from the model
        if hasattr(self, 'adapter_model') and hasattr(self.adapter_model.model, 'model'):
            embed_fn = self.adapter_model.model.model.embed_tokens
        else:
            embed_fn = self.llm_model.model.embed_tokens

        text_embeds = embed_fn(tokenized_text.input_ids)

        bos = torch.ones([tokenized_text.input_ids.shape[0], 1],
                         dtype=tokenized_text.input_ids.dtype,
                         device=device) * self.tokenizer.bos_token_id

        bos_embeds = embed_fn(bos)
        input_embeds = torch.cat([bos_embeds, img_proj, text_embeds], dim=1)
        
        # Generate with the appropriate model
        with torch.no_grad():
            if hasattr(self, 'adapter_model'):
                outputs = self.adapter_model.generate(
                    inputs_embeds=input_embeds,
                    max_length=max_length,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )
            else:
                outputs = self.llm_model.generate(
                    inputs_embeds=input_embeds,
                    max_length=max_length,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )
        
        return outputs

def parse_arguments():
    parser = argparse.ArgumentParser(description="Multimodal Medical Image + Hinglish Text Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint (.pt file)")
    parser.add_argument("--image", type=str, required=True, help="Path to the medical image")
    parser.add_argument("--text", type=str, required=True, help="Hinglish text input")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to run inference on (cuda/cpu)")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum length of generated text")
    parser.add_argument("--output", type=str, default="summary.txt", help="Path to save the generated summary")
    return parser.parse_args()

def load_image(image_path):
    transforms = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
    ])
    
    print(f"Loading and processing image: {image_path}")
    image = transforms(Image.open(image_path).convert("RGB"))
    
    # Ensure image has 3 channels (some medical images might be grayscale)
    if image.shape[0] < 3:
        image = torch.cat([image, image, image], dim=0)
    else:
        image = image[:3, :, :]
        
    return image.unsqueeze(0)  # Add batch dimension

def main():
    args = parse_arguments()
    device = torch.device(args.device)
    
    # Check if files exist
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image file not found: {args.image}")
    
    print(f"Using device: {device}")
    
    # Initialize the model
    model_name = "HuggingFaceH4/zephyr-7b-alpha"
    model = MultimodalLLM(model_name)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    
    # Print some keys from checkpoint for debugging
    print("Sample checkpoint keys:")
    keys = list(checkpoint.keys())
    print(keys[:5])  # Print first 5 keys
    
    # Load with custom state_dict loader to handle key mismatches
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model.eval()
    
    print("Is model on CUDA?", next(model.parameters()).is_cuda)
    
    # Process image
    image = load_image(args.image).to(device)
    
    # Format input text
    hinglish_text = args.text
    input_text = f"{hinglish_text}SUMMARY: "
    
    print(f"Input text: {input_text}")
    
    # Generate summary
    print("Generating summary...")
    with torch.no_grad():
        outputs = model.generate(
            text=input_text,
            image=image,
            max_length=args.max_length
        )
    
    # Decode the outputs
    summary = model.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    # Extract the part after "SUMMARY:" if present
    if "SUMMARY:" in summary:
        summary = summary.split("SUMMARY:")[1].strip()
    
    # Print and save the summary
    print("\nGenerated Summary:")
    print("-" * 50)
    print(summary)
    print("-" * 50)
    
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(summary)
    
    print(f"Summary saved to {args.output}")

if __name__ == "__main__":
    main()
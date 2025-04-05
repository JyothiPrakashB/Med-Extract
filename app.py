#!/usr/bin/env python3
"""
Usage:
    python app.py --checkpoint /path/to/checkpoint.pt
"""

import os
import argparse
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
try:
    import gradio as gr
    import pkg_resources
    gradio_version = pkg_resources.get_distribution("gradio").version
    print(f"Using Gradio version: {gradio_version}")
except ImportError:
    print("Gradio not installed. Please install it using:")
    print("pip install gradio")
    exit(1)


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
        
        # a new state dict with remapped keys
        new_state_dict = {}
        
        # copy of all keys that exactly match
        model_keys = dict(self.named_parameters())
        for key in state_dict:
            if key in model_keys:
                new_state_dict[key] = state_dict[key]
        
        # special cases for lm_head weights
        if "llm_model.lm_head.weight" in state_dict and "llm_model.lm_head.base_layer.weight" in model_keys:
            new_state_dict["llm_model.lm_head.base_layer.weight"] = state_dict["llm_model.lm_head.weight"]
        
        if "kbit_model.lm_head.weight" in state_dict and "kbit_model.lm_head.base_layer.weight" in model_keys:
            new_state_dict["kbit_model.lm_head.base_layer.weight"] = state_dict["kbit_model.lm_head.weight"]
            
        if "adapter_model.base_model.model.lm_head.weight" in state_dict and "adapter_model.base_model.model.lm_head.base_layer.weight" in model_keys:
            new_state_dict["adapter_model.base_model.model.lm_head.base_layer.weight"] = state_dict["adapter_model.base_model.model.lm_head.weight"]
        
        # For LoRA keys, check the update of patterns
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
        
        # Get the correct embed_tokens function from the model
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

    def generate(self, text, image, max_new_tokens=128):
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
        
        # Calculate the input sequence length
        sequence_length = input_embeds.shape[1]
        
        # Generate with the appropriate model
        with torch.no_grad():
            if hasattr(self, 'adapter_model'):
                outputs = self.adapter_model.generate(
                    inputs_embeds=input_embeds,
                    max_new_tokens=max_new_tokens,  # Use max_new_tokens instead of max_length
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            else:
                outputs = self.llm_model.generate(
                    inputs_embeds=input_embeds,
                    max_new_tokens=max_new_tokens,  # Use max_new_tokens instead of max_length
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
        
        # Decode the outputs
        summary = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        
        # Extract the part after "SUMMARY:" if present
        if "SUMMARY:" in summary:
            summary = summary.split("SUMMARY:")[1].strip()
            
        return summary


def load_image_for_model(image_path):
    """Load and preprocess an image for model input"""
    transforms = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
    ])
    
    image = transforms(Image.open(image_path).convert("RGB"))
    
    # Ensure image has 3 channels (some medical images might be grayscale)
    if image.shape[0] < 3:
        image = torch.cat([image, image, image], dim=0)
    else:
        image = image[:3, :, :]
        
    return image.unsqueeze(0)  # Add batch dimension

def parse_arguments():
    parser = argparse.ArgumentParser(description="Multimodal Medical Image + Hinglish Text Inference with Gradio Interface")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint (.pt file)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to run inference on (cuda/cpu)")
    parser.add_argument("--share", action="store_true", help="Create a public link for the interface")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the Gradio interface on")
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Check if checkpoint file exists
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Initialize the model
    model_name = "HuggingFaceH4/zephyr-7b-alpha"
    model = MultimodalLLM(model_name)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    
    # Load with custom state_dict loader to handle key mismatches
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model.eval()
    
    print("Is model on CUDA?", next(model.parameters()).is_cuda)
    
    # Define inference function for Gradio
    def predict(image, hinglish_text, max_tokens):
        try:
            # Process the input image
            processed_image = load_image_for_model(image)
            processed_image = processed_image.to(device)
            
            # Format input text
            input_text = f"{hinglish_text}SUMMARY: "
            
            # Generate summary
            with torch.no_grad():
                summary = model.generate(
                    text=input_text,
                    image=processed_image,
                    max_new_tokens=max_tokens
                )
                
            return summary
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    # Create the simplest possible Gradio interface that should work across all versions
    demo = gr.Interface(
        fn=predict,
        inputs=[
            gr.Image(type="filepath", label="Upload Medical Image"),
            gr.Textbox(label="Hinglish Text Input", placeholder="Enter your Hinglish text here..."),
            gr.Slider(minimum=16, maximum=256, value=128, step=16, label="Maximum Output Tokens")
        ],
        outputs=gr.Textbox(label="Generated English Summary"),
        title="Medical Image + Hinglish Text Analysis",
        description="Upload a medical image and enter Hinglish text to generate an English summary.",
        article="""
        ## Example Usage
        1. Upload a medical image (X-ray, MRI, CT scan, etc.)
        2. Enter your Hinglish text describing the symptoms or medical condition
        3. Adjust the maximum output tokens if needed
        4. Click 'Submit' to get an English summary
        """
    )
        
    # Try a range of ports
    try:
        demo.launch(share=args.share, server_port=args.port)
    except OSError:
        print("Default port is busy, trying alternate ports...")
        # Try ports 7861-7870
        for alt_port in range(7861, 7871):
            try:
                print(f"Trying port {alt_port}...")
                demo.launch(share=args.share, server_port=alt_port)
                print(f"Successfully launched on port {alt_port}")
                break
            except OSError:
                continue
        else:
            print("Could not find an available port. Please specify a free port using --port argument.")
            print("Example: python app.py --checkpoint my_checkpoint.pt --port 8000")

if __name__ == "__main__":
    main()
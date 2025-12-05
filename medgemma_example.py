"""
MedGemma VQA Inference Script
This script performs Visual Question Answering on medical images using Google's MedGemma model.
"""

import os
import json
import torch
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers import __version__ as transformers_version

# Suppress torch dynamo errors to fall back to eager execution
import torch._dynamo
torch._dynamo.config.suppress_errors = True

print(f"Transformers version: {transformers_version}")


def apply_transformers_workarounds():
    """Apply various workarounds for transformers compatibility issues"""
    
    # Workaround 1: ALL_PARALLEL_STYLES issue
    try:
        from transformers import modeling_utils
        if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
            modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none", "colwise", "rowwise"]
            print("Applied ALL_PARALLEL_STYLES workaround")
    except ImportError:
        pass
    
    # Workaround 2: Attention implementation mapping issue
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from transformers.models.gemma3.modeling_gemma3 import (
            Gemma3Attention,
            Gemma3SdpaAttention,
            Gemma3FlashAttention2,
        )
        
        # Ensure all attention implementations are properly mapped
        attention_mapping = {
            "eager": Gemma3Attention,
            "sdpa": Gemma3SdpaAttention,
            "flash_attention_2": Gemma3FlashAttention2,
        }
        
        for key, value in attention_mapping.items():
            if key not in ALL_ATTENTION_FUNCTIONS._global_mapping:
                ALL_ATTENTION_FUNCTIONS._global_mapping[key] = value
        
        print("Applied attention functions workaround")
        
    except (ImportError, AttributeError) as e:
        print(f"Could not apply attention workaround: {e}")
    
    # Workaround 3: Force specific attention implementation
    os.environ["TRANSFORMERS_ATTENTION_TYPE"] = "eager"


# Apply all workarounds before loading the model
apply_transformers_workarounds()


class MedGemmaVQAInference:
    """
    MedGemma Visual Question Answering Inference Engine
    
    This class handles loading the MedGemma model and processing medical VQA tasks.
    """
    
    def __init__(self, model_name="google/medgemma-4b-it", device="auto"):
        """
        Initialize the MedGemma model and processor for VQA tasks
        
        Args:
            model_name (str): Name or path of the model
            device (str): Device to run inference on ("auto", "cuda", or "cpu")
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        print(f"Using device: {self.device}")
        
        # Load model and processor
        self._load_model()

    def _load_model(self):
        """Load the model and processor with fallback options"""
        print('Loading Model and Processor...')
        
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="eager",  # Force eager attention to avoid compatibility issues
                trust_remote_code=True,
            )
            self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
            print("Model and processor loaded successfully")
            
        except Exception as e:
            print(f"Error loading model with eager attention: {e}")
            print("Trying alternative loading method...")
            
            # Fallback: try loading without specific attention implementation
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
            print("Model loaded with fallback method")

    def load_images(self, image_paths, base_path=""):
        """
        Load images from paths
        
        Args:
            image_paths (list): List of image paths
            base_path (str): Base path to prepend to image paths
            
        Returns:
            list: List of loaded PIL images (limited to 2 images)
        """
        images = []
        for img_path in image_paths:
            full_path = Path(base_path) / img_path.lstrip('/')
            
            # Handle both .dcm and .png formats
            if full_path.suffix == '.dcm':
                full_path = full_path.with_suffix('.png')
            
            try:
                img = Image.open(str(full_path)).convert('RGB')
                images.append(img)
            except Exception as e:
                print(f"Error loading image {full_path}: {e}")
        
        # Limit to 2 images for optimal performance
        return images[:2]

    def generate_prompt(self, question: str, options: list, context: str = "") -> str:
        """
        Generate a prompt for the medical VQA model
        
        Args:
            question (str): The medical question to be answered
            options (list): List of option strings
            context (str, optional): Additional context or patient information
            
        Returns:
            str: Formatted prompt string ready for model input
        """
        # Format the question and options as a dictionary
        try:
            formatted_options = {
                opt.strip().split('.')[0].strip(): opt.strip().split('.', 1)[1].strip()
                for opt in options if '.' in opt
            }
        except Exception:
            # Fallback if options don't follow expected format
            formatted_options = {chr(65 + i): opt.strip() for i, opt in enumerate(options)}
        
        question_data = {
            "Question": question.strip(),
            "Options": formatted_options
        }
        
        context_text = f"Patient information: {context}. " if context else ""
        
        prompt = f"""{context_text}You are a medical expert assistant. Please analyze the provided medical image(s) and answer the multiple-choice question.

Please provide your response in JSON format as follows: {{"answer": "letter_of_correct_option", "explanation": "brief explanation of your choice"}}

Question and Options:
{json.dumps(question_data, indent=2)}

Please select the most appropriate answer and provide a brief medical explanation."""

        return prompt

    def process_single_case(self, case_data, base_path=""):
        """
        Process a single VQA case
        
        Args:
            case_data (dict): Case data including images, question, and options
            base_path (str): Base path for image loading
            
        Returns:
            dict: Results including model's answer and explanation
        """
        # Load images
        images = self.load_images(case_data['image_path_list'], base_path)
        if not images:
            return {"error": "No images could be loaded"}

        # Generate prompt
        prompt = self.generate_prompt(
            case_data['question'],
            case_data['options'],
            case_data.get('context', '')
        )

        # Create messages for MedGemma chat template
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are an expert medical AI assistant specializing in medical image analysis and diagnosis."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ] + [{"type": "image", "image": img} for img in images]
            }
        ]

        try:
            # Process inputs using chat template
            inputs = self.processor.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                tokenize=True,
                return_dict=True, 
                return_tensors="pt"
            ).to(self.model.device, dtype=torch.bfloat16)

            input_len = inputs["input_ids"].shape[-1]

            # Generate response
            with torch.inference_mode():
                generation = self.model.generate(
                    **inputs, 
                    max_new_tokens=300, 
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    temperature=0.0,
                )
                generation = generation[0][input_len:]

            # Decode the generated text
            response = self.processor.decode(generation, skip_special_tokens=True).strip()

            # Clean up to free GPU memory
            del inputs, generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return {
                "model_response": response,
                "question_id": case_data.get('study_id', '') + '_' + case_data.get('task_name', ''),
                "question": case_data.get('question', ''),
                "options": case_data.get('options', []),
                "correct_answer": case_data.get('correct_answer', ''),
                "category": case_data.get('category', ''),
                "subcategory": case_data.get('subcategory', ''),
                "context": case_data.get('context', '')
            }

        except Exception as e:
            return {"error": f"Processing error: {str(e)}"}

    def process_batch(self, json_data, base_path="", output_file="results.json"):
        """
        Process multiple cases with progress bar and checkpointing
        
        Args:
            json_data (dict): Dictionary containing multiple cases
            base_path (str): Base path for image loading
            output_file (str): Path to save results
            
        Returns:
            dict: Results for all cases
        """
        # Load existing results if available
        results = {}
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r') as f:
                    results = json.load(f)
                # Remove all items with errors to retry them
                results = {k: v for k, v in results.items() if 'error' not in v}
                print(f"Loaded {len(results)} existing results from {output_file}")
            except json.JSONDecodeError:
                print(f"Error loading existing results from {output_file}, starting fresh")

        # Create progress bar
        pbar = tqdm(total=len(json_data), desc="Processing VQA cases")
        pbar.update(len(results))

        # Process remaining cases
        for case_id, case_data in json_data.items():
            # Skip if already processed
            if case_id in results:
                continue

            try:
                results[case_id] = self.process_single_case(case_data, base_path)
                
                # Save results after each successful case (checkpointing)
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
                    
                # Print errors for debugging
                if "error" in results[case_id]:
                    print(f"\nError processing {case_id}: {results[case_id]['error']}")
                
            except Exception as e:
                results[case_id] = {"error": str(e)}
                # Also save on error
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"\nException processing {case_id}: {str(e)}")

            pbar.update(1)

        pbar.close()
        return results


def main():
    """Main function to run MedGemma VQA inference"""
    import argparse
    
    parser = argparse.ArgumentParser(description='MedGemma VQA Inference')
    parser.add_argument('--input_file', type=str, required=True,
                       help='Input JSON file with VQA cases')
    parser.add_argument('--output_file', type=str, required=True,
                       help='Output JSON file for results')
    parser.add_argument('--base_path', type=str, default="",
                       help='Base path for image loading')
    parser.add_argument('--model_name', type=str, 
                       default='google/medgemma-4b-it',
                       help='Model name or path')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Initialize model
    print("Initializing MedGemma VQA model...")
    inferencer = MedGemmaVQAInference(model_name=args.model_name)
    
    # Load JSON data
    print(f"Loading VQA cases from {args.input_file}")
    with open(args.input_file, 'r') as f:
        cases = json.load(f)
    
    print(f"Found {len(cases)} VQA cases to process")
    
    # Process all cases with progress bar and checkpointing
    results = inferencer.process_batch(cases, args.base_path, args.output_file)
    
    print(f"\nProcessing complete. Results saved to {args.output_file}")
    print(f"Successfully processed {len([r for r in results.values() if 'error' not in r])} cases")
    print(f"Errors in {len([r for r in results.values() if 'error' in r])} cases")


if __name__ == "__main__":
    main()
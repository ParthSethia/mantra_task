#!/usr/bin/env python3
"""
Quick test script for Florence-2 model compatibility
"""

import sys
import torch
from PIL import Image
import numpy as np

def test_florence2_loading():
    """Test if Florence-2 can be loaded successfully"""
    print("Testing Florence-2 model loading...")
    print(f"PyTorch version: {torch.__version__}")
    
    try:
        from transformers import AutoProcessor, AutoModelForCausalLM
        print("✓ Transformers import successful")
        
        model_name = "microsoft/Florence-2-large"
        print(f"Loading model: {model_name}")
        
        # Test processor loading
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        print("✓ Processor loaded successfully")
        
        # Test model loading with fallback strategies
        model = None
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Target device: {device}")
        
        try:
            print("Attempting modern loading...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float32,
                attn_implementation="eager"
            )
            print("✓ Modern loading successful")
        except Exception as e:
            print(f"Modern loading failed: {e}")
            
            try:
                print("Attempting fallback loading...")
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float32
                )
                print("✓ Fallback loading successful")
            except Exception as e2:
                print(f"Fallback loading failed: {e2}")
                
                try:
                    print("Attempting basic loading...")
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        trust_remote_code=True
                    )
                    print("✓ Basic loading successful")
                except Exception as e3:
                    print(f"✗ All loading strategies failed: {e3}")
                    return False
        
        if model:
            print("Moving model to device...")
            model.to(device)
            print("✓ Model loaded and moved to device successfully!")
            
            # Test with a simple image
            print("Testing inference...")
            
            # Create a simple test image
            test_image = Image.new('RGB', (224, 224), color='red')
            
            task = "<DETAILED_CAPTION>"
            inputs = processor(text=task, images=test_image, return_tensors="pt").to(device)
            
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=50,
                    num_beams=1,
                    do_sample=False
                )
            
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed_answer = processor.post_process_generation(
                generated_text, task=task, image_size=(224, 224)
            )
            
            print(f"✓ Test inference successful: {parsed_answer}")
            return True
        
        return False
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        print("Please install: pip install transformers torch")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def suggest_alternatives():
    """Suggest alternative approaches if Florence-2 fails"""
    print("\n" + "="*50)
    print("FLORENCE-2 ALTERNATIVES")
    print("="*50)
    print("If Florence-2 continues to fail, consider these options:")
    print()
    print("1. Use BLIP instead:")
    print("   Change config.yaml: model_type: 'blip'")
    print("   Model: Salesforce/blip-image-captioning-large")
    print()
    print("2. Stick with Ollama (most reliable):")
    print("   Change config.yaml: model_type: 'ollama'")
    print()
    print("3. Try a different Florence-2 version:")
    print("   - microsoft/Florence-2-base (smaller, might work better)")
    print("   - microsoft/Florence-2-base-ft (fine-tuned version)")
    print()
    print("4. Update transformers version:")
    print("   pip install transformers==4.36.2")
    print("   # Some users report this specific version works better")

if __name__ == "__main__":
    print("Florence-2 Compatibility Test")
    print("=" * 40)
    
    success = test_florence2_loading()
    
    if success:
        print("\n✓ Florence-2 is working correctly!")
        print("You can now use Florence-2 in your video analysis.")
    else:
        print("\n✗ Florence-2 failed to load properly.")
        suggest_alternatives()
        sys.exit(1)
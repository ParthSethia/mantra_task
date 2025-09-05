import torch
from PIL import Image
from typing import List
import yaml

class HuggingFaceVLM:
    """Simple HuggingFace VLM wrapper for BLIP or Florence-2"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.vlm_config = config['vlm']
        self.hf_config = self.vlm_config['huggingface']
        self.model_type = self.vlm_config['model_type']
        
        self.device = self._get_device()
        self.processor = None
        self.model = None
        
        if self.model_type in ['blip', 'florence2']:
            self._load_model()
    
    def _get_device(self) -> str:
        device_config = self.hf_config.get('device', 'auto')
        
        if device_config == 'auto':
            if torch.cuda.is_available():
                return 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 'mps'
            else:
                return 'cpu'
        return device_config
    
    def _load_model(self):
        """Load the specified HuggingFace model"""
        try:
            if self.model_type == 'blip':
                from transformers import BlipProcessor, BlipForConditionalGeneration
                
                model_name = self.hf_config['blip_model']
                print(f"Loading BLIP model: {model_name}")
                
                self.processor = BlipProcessor.from_pretrained(model_name)
                self.model = BlipForConditionalGeneration.from_pretrained(model_name)
                
            elif self.model_type == 'florence2':
                from transformers import AutoProcessor, AutoModelForCausalLM
                
                model_name = self.hf_config['florence2_model']
                print(f"Loading Florence-2 model: {model_name}")
                
                self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
                
                # Load model with fallback strategies for Florence-2 compatibility
                try:
                    # Try with modern settings first
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        trust_remote_code=True,
                        torch_dtype=torch.float32,
                        attn_implementation="eager"
                    )
                except (TypeError, AttributeError) as e:
                    print(f"Modern loading failed ({e}), trying fallback...")
                    # Fallback for older transformers versions
                    try:
                        self.model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            trust_remote_code=True,
                            torch_dtype=torch.float32
                        )
                    except Exception as e2:
                        print(f"Fallback loading failed ({e2}), trying basic load...")
                        # Last resort - basic loading
                        self.model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            trust_remote_code=True
                        )
            
            self.model.to(self.device)
            print(f"Model loaded on {self.device}")
            
        except ImportError:
            raise ImportError("transformers library required: pip install transformers torch")
        except Exception as e:
            raise Exception(f"Failed to load {self.model_type} model: {e}")
    
    def analyze_image(self, image: Image.Image) -> str:
        """Analyze a single image"""
        if self.model_type == 'ollama':
            raise ValueError("Use Ollama directly for ollama model_type")
        
        try:
            if self.model_type == 'blip':
                return self._blip_analyze(image)
            elif self.model_type == 'florence2':
                return self._florence2_analyze(image)
        except Exception as e:
            return f"Analysis failed: {str(e)}"
    
    def _blip_analyze(self, image: Image.Image) -> str:
        """BLIP image analysis"""
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        
        max_length = self.hf_config.get('blip_max_length', 50)
        
        with torch.no_grad():
            out = self.model.generate(**inputs, max_length=max_length, num_beams=3)
        
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption
    
    def _florence2_analyze(self, image: Image.Image) -> str:
        """Florence-2 image analysis"""
        task = self.hf_config.get('florence2_task', '<DETAILED_CAPTION>')
        
        inputs = self.processor(text=task, images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3
            )
        
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        
        parsed_answer = self.processor.post_process_generation(
            generated_text, task=task, image_size=(image.width, image.height)
        )
        
        return str(parsed_answer)
    
    def analyze_batch(self, images: List[Image.Image]) -> List[str]:
        """Analyze multiple images"""
        return [self.analyze_image(img) for img in images]
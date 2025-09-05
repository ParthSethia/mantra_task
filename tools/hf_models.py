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
        
        if image is None:
            return "Analysis failed: Image not available"
            
        try:
            if self.model_type == 'blip':
                return self._blip_analyze(image)
            elif self.model_type == 'florence2':
                return self._florence2_analyze(image)
            else:
                return f"Unknown model type: {self.model_type}"
            
        except Exception as e:
            return f"Analysis failed: {str(e)}"
    
    def _blip_analyze(self, image: Image.Image) -> str:
        """BLIP image analysis"""
        if image is None:
            return "Analysis failed: Image is None"
            
        try:
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            
            max_length = self.hf_config.get('blip_max_length', 50)
            
            with torch.no_grad():
                out = self.model.generate(**inputs, max_length=max_length, num_beams=3)
            
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            return caption
            
        except Exception as e:
            return f"Analysis failed: {str(e)}"
        
    def _florence2_analyze(self, image: Image.Image) -> str:
        """Florence-2 image analysis with robust fallbacks"""
        if image is None:
            return "Analysis failed: Image is None"

        try:
            if not hasattr(image, 'width') or not hasattr(image, 'height'):
                return "Analysis failed: Image missing width/height attributes"

            task = self.hf_config.get('florence2_task', '<DETAILED_CAPTION>')
            inputs = self.processor(text=task, images=image, return_tensors="pt").to(self.device)
            
            generated_ids = None

            # Strategy 1: Standard generation
            try:
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=512,
                        do_sample=False
                    )
            except Exception:
                # Strategy 2: Disable cache (fixes Florence-2 past_key_values issue)
                try:
                    with torch.no_grad():
                        generated_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=512,
                            do_sample=False,
                            use_cache=False
                        )
                except Exception:
                    # Strategy 3: Manual forward pass fallback
                    try:
                        with torch.no_grad():
                            outputs = self.model(**inputs)
                            logits = outputs.logits
                            tokens = torch.argmax(logits, dim=-1)
                            generated_text = self.processor.batch_decode(tokens, skip_special_tokens=True)[0]
                        return generated_text
                    except Exception as e3:
                        raise Exception(f"All Florence-2 generation strategies failed. Last error: {e3}")

            if generated_ids is None:
                raise Exception("No generation strategy succeeded")

            # Decode and post-process
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed_answer = self.processor.post_process_generation(
                generated_text, task=task, image_size=(image.width, image.height)
            )

            return str(parsed_answer)

        except Exception as e:
            return f"Analysis failed: {str(e)}"

    
    def analyze_batch(self, images: List[Image.Image]) -> List[str]:
        """Analyze multiple images"""
        results = []
        for img in images:
            if img is None:
                results.append("Analysis failed: Image not available")
            else:
                results.append(self.analyze_image(img))
        return results
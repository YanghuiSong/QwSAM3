# core/qwen_agent.py
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import json
import re
from config import Config

class QwenAgent:
    """Qwen3-VL Intelligent Agent - English Version"""
    
    def __init__(self, model_path=Config.QWEN_MODEL_PATH, device=None):
        print("Loading Qwen3-VL model...")
        
        # Get device string
        if device is None:
            self.device_str = Config.get_qwen_device()
        else:
            self.device_str = device
        
        print(f"  Model path: {model_path}")
        print(f"  Device: {self.device_str}")
        
        try:
            # Get data type
            torch_dtype_str = Config.get_qwen_dtype()
            torch_dtype = getattr(torch, torch_dtype_str)
            
            print(f"  Model precision: torch.{torch_dtype_str}")
            
            # Load model
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map=self.device_str if self.device_str != "cpu" else None,
                trust_remote_code=True
            )
            
            # If device is CPU, move model manually
            if self.device_str == "cpu":
                self.model = self.model.to("cpu")
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            print(f"✓ Qwen3-VL model loaded successfully")
            print(f"  Model device: {self.model.device}")
            
        except Exception as e:
            print(f"✗ Model loading failed: {e}")
            print("Trying alternative loading methods...")
            self._load_with_fallback(model_path)
        
        self.model.eval()
    
    def _load_with_fallback(self, model_path):
        """Alternative loading methods"""
        try:
            print("Trying without device_map...")
            
            # Get data type
            torch_dtype_str = Config.get_qwen_dtype()
            torch_dtype = getattr(torch, torch_dtype_str)
            
            # Method 1: Load to CPU first then move
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map=None,
                trust_remote_code=True
            )
            
            # Move to specified device
            if self.device_str != "cpu":
                self.model = self.model.to(self.device_str)
            
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            print("✓ Alternative method 1 successful")
            
        except Exception as e:
            print(f"Alternative method 1 failed: {e}")
            
            try:
                print("Trying with lower precision...")
                # Method 2: Use FP32 to reduce memory
                self.model = AutoModelForVision2Seq.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32,
                    device_map=None,
                    trust_remote_code=True
                )
                
                # Move to specified device
                if self.device_str != "cpu":
                    self.model = self.model.to(self.device_str)
                
                self.processor = AutoProcessor.from_pretrained(
                    model_path,
                    trust_remote_code=True
                )
                
                print("✓ Alternative method 2 successful")
                
            except Exception as e:
                print(f"Alternative method 2 failed: {e}")
                raise RuntimeError("All loading methods failed")
    
    def analyze_scene(self, image_path, detail_level="high"):
        """
        Detailed scene analysis for full segmentation
        
        Args:
            image_path: Image path
            detail_level: Detail level (low, medium, high)
            
        Returns:
            Scene analysis result in English
        """
        try:
            # Load image
            if isinstance(image_path, str):
                image = Image.open(image_path).convert("RGB")
            else:
                image = image_path
            
            # Select prompt based on detail level
            if detail_level == "high":
                prompt = """Please analyze this image in detail as a professional image analyst.
                
                Please describe in detail following this structure:
                
                1. Main Object Identification:
                   - List all visible main objects (at least 10)
                   - Describe each object's position, size, color, shape
                
                2. Scene Composition:
                   - Foreground, midground, background division
                   - Spatial layout and perspective
                
                3. Object Categories:
                   - Person category (e.g., person, child, adult)
                   - Vehicle category (e.g., car, bicycle, motorcycle)
                   - Building category (e.g., building, house, shop)
                   - Nature category (e.g., tree, plant, sky, cloud)
                   - Other objects (e.g., furniture, equipment, signs)
                
                4. Special Attention:
                   - Occlusion relationships
                   - Repeated objects
                   - Small or subtle objects
                
                Please answer in English. Description must be extremely detailed and specific."""
            elif detail_level == "medium":
                prompt = """Please analyze this image and list all visible objects and regions. Answer in English."""
            else:
                prompt = """Please briefly describe this image content. Answer in English."""
            
            # Build messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            text = self.processor.apply_chat_template(
                messages, 
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Prepare inputs
            inputs = self.processor(
                text=text,
                images=image,
                return_tensors="pt"
            )
            
            # Move inputs to model device
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate analysis
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=2000,  # Increase tokens
                    temperature=0.1,
                    do_sample=False
                )
            
            # Decode output
            generated_text = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0]
            
            # Extract assistant response
            if "assistant" in generated_text:
                response = generated_text.split("assistant")[-1].strip()
            else:
                response = generated_text
            
            response = response.replace("<|im_end|>", "").strip()
            
            print(f"Scene analysis completed, length: {len(response)} characters")
            
            return response
            
        except Exception as e:
            print(f"Scene analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return f"Scene analysis failed: {str(e)}"
    
    def describe_image(self, image_path, max_tokens=500):
        """Let Qwen3 describe image content in English"""
        try:
            # Load image
            if isinstance(image_path, str):
                image = Image.open(image_path).convert("RGB")
            else:
                image = image_path
            
            # Build conversation message
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": "Please describe this image in detail in English, including main objects, their relative positions, quantities, etc. "
                         "Description should be specific and detailed to facilitate subsequent image segmentation."}
                    ]
                }
            ]
            
            # Prepare input
            text = self.processor.apply_chat_template(
                messages, 
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Encode input
            inputs = self.processor(
                text=text,
                images=image,
                return_tensors="pt"
            )
            
            # Move inputs to model device
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate description
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    temperature=0.1
                )
            
            # Decode output
            generated_text = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0]
            
            # Extract assistant response
            if "assistant" in generated_text:
                response = generated_text.split("assistant")[-1].strip()
            else:
                response = generated_text
            
            # Clean response
            response = response.replace("<|im_end|>", "").strip()
            
            return response
            
        except Exception as e:
            print(f"Image description failed: {e}")
            import traceback
            traceback.print_exc()
            return f"Image description generation failed: {str(e)}"
    
    def generate_instructions(self, image_path, user_request, image_description=None):
        """Generate SAM3 segmentation instruction sequence in English"""
        try:
            # Load image
            if isinstance(image_path, str):
                image = Image.open(image_path).convert("RGB")
            else:
                image = image_path
            
            # If no description, generate one
            if image_description is None:
                print("Generating image description...")
                image_description = self.describe_image(image)
                print(f"Image description generated ({len(image_description)} characters)")
            
            # Build system prompt
            system_prompt = Config.INSTRUCTION_TEMPLATE.format(
                image_description=image_description[:500],  # Limit description length
                user_request=user_request
            )
            
            # Build messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": system_prompt}
                    ]
                }
            ]
            
            text = self.processor.apply_chat_template(
                messages, 
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Prepare inputs
            inputs = self.processor(
                text=text,
                images=image,
                return_tensors="pt"
            )
            
            # Move inputs to model device
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            print("Generating segmentation instructions...")
            # Generate instructions
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=1500,  # Increase tokens
                    temperature=0.1,  # Lower temperature for more focused output
                    do_sample=False,   # Deterministic output
                    repetition_penalty=1.1
                )
            
            # Decode
            generated_text = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0]
            
            # Extract JSON part
            instructions = self._extract_json_from_response(generated_text)
            
            # Validate instruction format
            if self._validate_instructions(instructions):
                print("✓ Instructions generated successfully")
                print(f"Strategy: {instructions.get('strategy', 'N/A')[:100]}...")
                print(f"Number of instructions: {len(instructions.get('instructions', []))}")
                return instructions
            else:
                print("⚠ Instruction format validation failed, using fallback instructions")
                return self._get_fallback_instructions(image_description, user_request)
            
        except Exception as e:
            print(f"Instruction generation failed: {e}")
            import traceback
            traceback.print_exc()
            return self._get_fallback_instructions("", user_request)
    
    def _extract_json_from_response(self, response):
        """Extract JSON from model response, handle incomplete JSON"""
        import json
        import re
        
        # First try to find complete JSON
        json_pattern = r'\{[\s\S]*?\}'
        matches = re.findall(json_pattern, response, re.DOTALL)
        
        if matches:
            # Try each match, find parseable JSON
            for json_str in reversed(matches):  # Try from last
                try:
                    parsed = json.loads(json_str)
                    return parsed
                except json.JSONDecodeError as e:
                    # Try to fix and complete JSON
                    fixed_json = self._fix_incomplete_json(json_str)
                    if fixed_json:
                        try:
                            parsed = json.loads(fixed_json)
                            return parsed
                        except Exception as e2:
                            continue
        
        # If no complete JSON found, construct from response
        return self._construct_json_from_response(response)
    
    def _fix_incomplete_json(self, json_str):
        """Fix incomplete JSON string"""
        import re
        
        # Remove extra newlines and spaces
        json_str = json_str.strip()
        
        # Complete missing closing symbols
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        open_brackets = json_str.count('[')
        close_brackets = json_str.count(']')
        
        result = json_str
        
        # Complete braces
        if open_braces > close_braces:
            result += '}' * (open_braces - close_braces)
        
        # Complete brackets
        if open_brackets > close_brackets:
            result += ']' * (open_brackets - close_brackets)
        
        # Remove trailing commas
        result = re.sub(r',\s*([}\]])', r'\1', result)
        result = re.sub(r',\s*$', '', result)
        
        # Ensure string ends with closing bracket
        if not result.endswith('}') and not result.endswith(']'):
            if '{' in result and '}' not in result:
                result += '}'
        
        return result
    
    def _construct_json_from_response(self, response):
        """Construct JSON structure from response text"""
        import re
        
        # Extract strategy description
        strategy_match = re.search(r'["\']?strategy["\']?\s*:\s*["\']([^"\']+)["\']', response, re.IGNORECASE)
        strategy = strategy_match.group(1) if strategy_match else "Segmentation based on image description"
        
        # Extract object keywords
        object_keywords = []
        
        # Common object keywords in English
        common_objects = [
            # Person category
            "person", "people", "human", "man", "woman", "child", "baby", "face",
            # Vehicle category
            "car", "vehicle", "truck", "bus", "motorcycle", "bicycle", "van", "train", "plane",
            # Building category
            "building", "house", "structure", "architecture", "shop", "store", "home", "wall",
            # Nature category
            "tree", "plant", "vegetation", "flower", "grass", "bush", "leaf", "sky", "cloud", "water",
            # Other objects
            "animal", "dog", "cat", "bird", "horse", "furniture", "chair", "table", "sofa", 
            "food", "fruit", "vegetable", "clothing", "shirt", "pants", "shoe", 
            "electronic", "computer", "phone", "television", "book", "paper"
        ]
        
        # Extract possible objects from response
        words = re.findall(r'\b\w{3,}\b', response.lower())
        for word in words:
            if word in common_objects and word not in object_keywords:
                object_keywords.append(word)
        
        # If not found, use default
        if not object_keywords:
            object_keywords = ["object", "thing", "item"]
        
        # Build instructions - prioritize text prompts
        instructions = []
        for i, obj in enumerate(object_keywords[:8]):  # Max 8 objects
            instructions.append({
                "type": "text",
                "content": obj
            })
        
        return {
            "strategy": strategy,
            "instructions": instructions
        }
    
    def _validate_instructions(self, instructions):
        """Validate instruction format"""
        if not instructions:
            return False
        
        required_keys = ["strategy", "instructions"]
        
        # Check top-level keys
        if not all(key in instructions for key in required_keys):
            return False
        
        # Check instructions array
        if not isinstance(instructions["instructions"], list):
            return False
        
        # Check each instruction
        for i, instr in enumerate(instructions["instructions"]):
            if "type" not in instr:
                return False
            
            if "content" not in instr:
                return False
            
            # Check if type is valid
            if instr["type"] not in ["text", "box", "point"]:
                return False
        
        return True
    
    def _get_fallback_instructions(self, image_description, user_request):
        """Generate fallback instructions in English"""
        import re
        
        # Extract possible object keywords
        english_words = re.findall(r'\b([a-z]{3,10})\b', image_description.lower())
        
        # Priority object mapping
        priority_map = {
            "person": ["person", "people", "human", "man", "woman"],
            "vehicle": ["car", "vehicle", "truck", "bus", "motorcycle", "bicycle"],
            "building": ["building", "house", "structure", "architecture"],
            "nature": ["tree", "plant", "vegetation", "flower", "grass"],
            "furniture": ["chair", "table", "sofa", "bed", "desk"],
            "animal": ["dog", "cat", "bird", "horse", "animal"]
        }
        
        # Extract from image description
        detected_objects = []
        for category, words in priority_map.items():
            for word in words:
                if word in english_words and word not in detected_objects:
                    detected_objects.append(word)
        
        # If no objects detected, use common ones
        if not detected_objects:
            detected_objects = ["person", "car", "building", "tree"]
        
        # Build instructions
        instructions = []
        for obj in detected_objects[:6]:  # Max 6 objects
            instructions.append({
                "type": "text",
                "content": obj
            })
        
        return {
            "strategy": "Fallback: Text prompts for common objects",
            "instructions": instructions
        }
    
    def extract_object_categories(self, scene_analysis, max_categories=15):
        """Extract object categories from scene analysis for full segmentation"""
        import re
        
        # Common object vocabulary in English
        common_objects = [
            "person", "people", "human", "man", "woman", "child", "baby",
            "car", "vehicle", "truck", "bus", "motorcycle", "bicycle", "van",
            "building", "house", "structure", "architecture", "shop", "store",
            "tree", "plant", "vegetation", "flower", "grass", "bush", "leaf",
            "road", "street", "pavement", "sidewalk", "path", "lane",
            "sky", "cloud", "sun", "moon", "air",
            "water", "river", "lake", "sea", "ocean", "pool",
            "ground", "floor", "earth", "land", "terrain", "field",
            "animal", "dog", "cat", "bird", "horse", "cow", "sheep",
            "furniture", "chair", "table", "sofa", "bed", "desk", "cabinet",
            "electronic", "computer", "phone", "television", "screen", "monitor",
            "food", "fruit", "vegetable", "meal", "drink", "coffee",
            "clothing", "shirt", "pants", "dress", "shoe", "hat", "jacket",
            "equipment", "machine", "tool", "device", "instrument",
            "book", "paper", "document", "magazine",
            "container", "box", "bag", "bottle", "cup", "glass", "plate",
            "light", "lamp", "bulb", "candle",
            "window", "door", "gate", "fence", "wall", "roof",
            "sign", "symbol", "logo", "label", "text"
        ]
        
        # Extract English words
        english_words = re.findall(r'\b([a-z]{3,10})\b', scene_analysis.lower())
        
        # Find matching objects
        detected_objects = []
        for word in english_words:
            if word in common_objects and word not in detected_objects:
                detected_objects.append(word)
        
        # Add generic categories if not enough
        if len(detected_objects) < max_categories // 2:
            generic_categories = [
                "object", "thing", "item", "element", "component",
                "region", "area", "zone", "section", "part",
                "material", "substance", "texture", "pattern",
                "shape", "form", "structure", "construction"
            ]
            for category in generic_categories:
                if len(detected_objects) < max_categories:
                    detected_objects.append(category)
        
        return detected_objects[:max_categories]
import yaml
import logging
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

class Domain:
    def __init__(self, id: str, name: str, description: str, response_style: str, 
                 example_questions: List[str] = None, keywords: List[str] = None, priority: int = 99):
        self.id = id
        self.name = name
        self.description = description
        self.response_style = response_style
        self.example_questions = example_questions or []
        self.keywords = keywords or []
        self.priority = priority
        self.require_disclaimer = False
        self.config = self

def get_default_domains() -> Dict[str, Domain]:
    """Get the base set of built-in domains"""
    return {
        "general": Domain(
            "general", "General Purpose", "Standard helper mode for general queries",
            "Helpful and conversational", ["What is AI?", "How does RAG work?"],
            priority=999
        ),
        "teaching_abc": Domain(
            "teaching_abc", "ABC Phonics Teacher", "Interactive phonics and reading tutor for children",
            "Child-friendly, encouraging, and phonetic", ["What sound does the letter A make?", "Help me read cat"],
            ["alphabet", "phonics", "reading", "letter", "sound"],
            priority=1
        ),
        "product_pricing": Domain(
            "product_pricing", "Product Pricing Assistant", "Lookup product specifications and prices",
            "Precise, structured, and financial", ["How much is a MacBook Air?", "What are the specs of Dell XPS?"],
            ["shopping", "best buy", "amazon", "price", "cost", "spec", "buy"],
            priority=2
        ),
        "medical": Domain(
            "medical", "Medical Advisor", "Retrieve medical information (with disclaimers)",
            "Clinical, objective, cautious", ["What are symptoms of flu?"],
            ["medical", "symptom", "flu", "health", "disease", "treatment", "fever", "remedy", "cure", "cough", "cold", "burn", "ache", "sick", "pain", "hurt"],
            priority=3
        ),
        "legal": Domain(
            "legal", "Legal Advisor", "Retrieve legal info and contract structures",
            "Formal, precise, structured", ["What is a NDA?"],
            ["legal", "contract", "nda", "law", "sue", "court"],
            priority=4
        ),
        "teaching_math": Domain(
            "teaching_math", "Math Teacher", "Simple math and arithmetic tutor",
            "Educational, step-by-step, patient", ["How do I add numbers?", "What is 2+2?"],
            ["math", "arithmetic", "addition", "subtraction", "multiply", "divide", "numbers", "add"],
            priority=5
        ),
        "sinhala_news": Domain(
            "sinhala_news", "Sinhala News Assistant", "News articles and classifications in Sinhala",
            "Informative news style", ["ලෝක පාපන්දු සම්මේලනයේ සභාපති කවුද?"],
            ["sinhala", "news", "ලංකා", "පුවත්", "ශ්‍රී ලංකා"],
            priority=7
        ),
        "survival": Domain(
            "survival", "Survival Guide Assistant", "Wilderness survival and emergency preparedness advice",
            "Practical, direct, clear", ["How to start a fire?", "How to find water?"],
            ["survival", "wilderness", "lost", "shelter", "fire", "distress", "signal", "bite", "water", "filter", "purify", "navigate", "navigation", "woods", "forest", "mountain", "desert"],
            priority=6
        )
    }

def load_domains_from_yaml(base_registry: Dict[str, Domain]) -> Dict[str, Domain]:
    """Load and update domain configurations dynamically from domains.yaml"""
    config_path = Path("domains.yaml")
    if not config_path.exists():
        logger.warning("domains.yaml not found. Using default domain settings.")
        return base_registry

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        enabled_list = config.get("enabled_domains", [])
        domains_cfg = config.get("domains", {})

        filtered_registry = {}
        
        # Keep general by default
        if "general" in base_registry:
            filtered_registry["general"] = base_registry["general"]

        for dom_id, dom in base_registry.items():
            if dom_id == "general":
                continue

            if dom_id in enabled_list:
                cfg = domains_cfg.get(dom_id, {}) or {}
                
                # Update priority
                if "priority" in cfg:
                    dom.priority = cfg["priority"]
                    
                # Merge custom_keywords
                custom_kws = cfg.get("custom_keywords", [])
                combined_kws = list(dom.keywords)
                for kw in custom_kws:
                    if kw not in combined_kws:
                        combined_kws.append(kw)
                dom.keywords = combined_kws
                
                if "require_disclaimer" in cfg:
                    dom.require_disclaimer = cfg["require_disclaimer"]
                    
                filtered_registry[dom_id] = dom
        
        # Support dynamic domain creation from custom_domains
        custom_cfg = config.get("custom_domains", {}) or {}
        for dom_id, cfg in custom_cfg.items():
            if dom_id in enabled_list and dom_id not in filtered_registry:
                name = cfg.get("name", dom_id.replace("_", " ").title())
                desc = cfg.get("description", "")
                style = cfg.get("response_style", "conversational")
                examples = cfg.get("example_questions", [])
                keywords = cfg.get("custom_keywords", [])
                priority = cfg.get("priority", 99)
                
                new_dom = Domain(
                    id=dom_id,
                    name=name,
                    description=desc,
                    response_style=style,
                    example_questions=examples,
                    keywords=keywords,
                    priority=priority
                )
                if "require_disclaimer" in cfg:
                    new_dom.require_disclaimer = cfg["require_disclaimer"]
                    
                filtered_registry[dom_id] = new_dom
                logger.info(f"Dynamically created new custom domain: {dom_id}")
                
        logger.info(f"Loaded and configured {len(filtered_registry)} domains dynamically from domains.yaml")
        return filtered_registry
    except Exception as e:
        logger.error(f"Failed to load domains.yaml: {e}")
        return base_registry

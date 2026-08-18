"""Utilities for online research and knowledge acquisition"""

import wikipedia
import logging
import requests
import yaml
from pathlib import Path
from typing import List, Dict

logger = logging.getLogger(__name__)

def clean_query(topic: str) -> str:
    import re
    # Lowercase and strip whitespace
    q = topic.lower().strip()
    # Remove question patterns at the beginning
    prefixes = [
        r"^do\s+you\s+know\s+what\s+is\s+a\s+", r"^do\s+you\s+know\s+what\s+is\s+",
        r"^do\s+you\s+know\s+who\s+is\s+", r"^do\s+you\s+know\s+who\s+are\s+",
        r"^do\s+you\s+know\s+", r"^can\s+you\s+tell\s+me\s+what\s+is\s+a\s+",
        r"^can\s+you\s+tell\s+me\s+what\s+is\s+", r"^can\s+you\s+tell\s+me\s+about\s+",
        r"^please\s+define\s+", r"^tell\s+me\s+what\s+is\s+a\s+", r"^tell\s+me\s+what\s+is\s+",
        r"^what\s+is\s+a\s+", r"^what\s+is\s+", r"^whats\s+is\s+a\s+", r"^whats\s+is\s+", 
        r"^what\s+are\s+", r"^who\s+is\s+", r"^who\s+are\s+", r"^define\s+", r"^tell\s+me\s+about\s+", 
        r"^explain\s+", r"^search\s+for\s+", r"^research\s+"
    ]
    for pattern in prefixes:
        q = re.sub(pattern, "", q)
        
    # Remove topic qualifiers at the end
    suffixes = [
        r"\s+in\s+science$", r"\s+as\s+per\s+science$", r"\s+definition$", r"\s+meaning$", r"\s+wikipedia$",
        r"\s+in\s+sceince$"
    ]
    for pattern in suffixes:
        q = re.sub(pattern, "", q)
        
    # Common spacing corrections
    q = q.replace("blackhole", "black hole")
    
    return q.strip("?.!, ")

class WebResearcher:
    """Handles online research to augment the knowledge base"""
    
    def __init__(self, language="en"):
        wikipedia.set_lang(language)
        wikipedia.set_user_agent("TinyLLM_RAG_Bot/1.0 (contact: support@example.com)")
    
    def research_wolfram(self, query: str) -> List[Dict]:
        """Query WolframAlpha Short Answers API to get exact math/science facts"""
        try:
            config_path = Path(__file__).parent.parent / "config.yaml"
            if config_path.exists():
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)
            else:
                config = {}
                
            wf_config = config.get("wolfram", {})
            appid = wf_config.get("appid", "YOUR_WOLFRAM_APPID")
            enabled = wf_config.get("enabled", True)
            
            if not enabled or not appid or appid == "YOUR_WOLFRAM_APPID":
                logger.warning("WolframAlpha research skipped: AppID not configured in config.yaml")
                return []
                
            logger.info(f"Querying WolframAlpha for: {query}")
            url = "https://api.wolframalpha.com/v1/result"
            params = {
                "appid": appid,
                "i": query
            }
            r = requests.get(url, params=params, timeout=5)
            if r.status_code == 200 and r.text:
                return [{
                    "text": f"WolframAlpha result for '{query}': {r.text}",
                    "source": "WolframAlpha",
                    "url": f"https://www.wolframalpha.com/input?i={requests.utils.quote(query)}"
                }]
        except Exception as e:
            logger.error(f"WolframAlpha query failed: {e}")
        return []

    def research_topic(self, topic: str, sentences: int = 5) -> List[Dict]:
        """Search Wikipedia/WolframAlpha for a topic and return structured items"""
        cleaned_topic = clean_query(topic)
        logger.info(f"Cleaned research topic: '{topic}' -> '{cleaned_topic}'")
        
        # Try WolframAlpha first if the query looks like math or scientific calculation
        math_keywords = ["+", "-", "*", "/", "=", "calculate", "solve", "math", "integrate", "derivative", "equation", "distance to", "speed of", "how far", "how many", "how much"]
        topic_lower = topic.lower()
        
        if any(kw in topic_lower for kw in math_keywords):
            wolfram_results = self.research_wolfram(cleaned_topic)
            if wolfram_results:
                return wolfram_results

        # Fallback to Wikipedia
        try:
            logger.info(f"Searching Wikipedia for: {cleaned_topic}")
            
            # Get search results
            search_results = wikipedia.search(cleaned_topic, results=3)
            if not search_results:
                return []
            
            findings = []
            for result in search_results:
                try:
                    # Get page summary
                    page = wikipedia.page(result, auto_suggest=False)
                    summary = page.summary
                    
                    # Split into meaningful chunks if needed, or just take summary
                    findings.append({
                        "text": summary,
                        "source": f"Wikipedia: {result}",
                        "url": page.url
                    })
                except (wikipedia.DisambiguationError, wikipedia.PageError):
                    continue
                    
            return findings
            
        except Exception as e:
            logger.error(f"Research failed for {cleaned_topic}: {e}")
            return []

    def get_quick_fact(self, query: str) -> str:
        """Get a single quick fact from the web"""
        try:
            return wikipedia.summary(query, sentences=2)
        except:
            return ""

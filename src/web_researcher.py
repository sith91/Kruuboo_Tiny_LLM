"""Utilities for online research and knowledge acquisition with configurable multi-provider fallback."""

import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional

import requests
import wikipedia
import yaml

logger = logging.getLogger(__name__)


def clean_query(topic: str) -> str:
    """Clean and normalize topic string before dispatching to research providers."""
    q = topic.lower().strip("?.!, \t\n")
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
        q = re.sub(pattern, "", q).strip("?.!, \t\n")

    suffixes = [
        r"\s+in\s+science$", r"\s+as\s+per\s+science$", r"\s+definition$", r"\s+meaning$", r"\s+wikipedia$",
        r"\s+in\s+sceince$"
    ]
    for pattern in suffixes:
        q = re.sub(pattern, "", q).strip("?.!, \t\n")

    q = q.replace("blackhole", "black hole")
    return q.strip("?.!, ")


class WebResearcher:
    """
    Handles online research with a resilient, multi-tiered fallback architecture.
    Configurable via config.yaml across providers: WolframAlpha, DuckDuckGo, Wikipedia, ArXiv.
    """

    clean_query = staticmethod(clean_query)

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.web_cfg = self.config.get("web_research", {})

        # Wikipedia setup
        wiki_cfg = self.web_cfg.get("wikipedia", {})
        language = wiki_cfg.get("language", "en")
        try:
            wikipedia.set_lang(language)
            wikipedia.set_user_agent("Kruuboo_SLM_Bot/1.0 (contact: support@example.com)")
        except Exception as e:
            logger.debug(f"Wikipedia initialization notice: {e}")

    def _load_config(self) -> Dict:
        """Load configuration from config.yaml with root path resolution."""
        for path in [self.config_path, Path(__file__).resolve().parent.parent / "config.yaml", Path("config.yaml")]:
            if path.exists():
                try:
                    with open(path, "r") as f:
                        return yaml.safe_load(f) or {}
                except Exception as e:
                    logger.debug(f"Could not load config from {path}: {e}")
        return {}

    def research_wolfram(self, query: str) -> List[Dict]:
        """Query WolframAlpha Short Answers API for exact math, physics, and computational facts."""
        wf_cfg = self.web_cfg.get("wolfram", self.config.get("wolfram", {}))
        appid = wf_cfg.get("appid", "YOUR_WOLFRAM_APPID")
        enabled = wf_cfg.get("enabled", True)
        timeout = wf_cfg.get("timeout", 5)

        if not enabled or not appid or appid == "YOUR_WOLFRAM_APPID":
            logger.debug("WolframAlpha research skipped: AppID not configured in config.yaml")
            return []

        try:
            logger.info(f"Querying WolframAlpha for: {query}")
            url = "https://api.wolframalpha.com/v1/result"
            params = {"appid": appid, "i": query}
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200 and r.text and "no short answer" not in r.text.lower():
                return [{
                    "text": f"WolframAlpha result for '{query}': {r.text.strip()}",
                    "source": "WolframAlpha",
                    "url": f"https://www.wolframalpha.com/input?i={requests.utils.quote(query)}"
                }]
        except Exception as e:
            logger.debug(f"WolframAlpha query failed: {e}")
        return []

    def research_duckduckgo(self, query: str) -> List[Dict]:
        """Query DuckDuckGo Instant Answer API for rapid unauthenticated web summaries."""
        ddg_cfg = self.web_cfg.get("duckduckgo", {})
        if not ddg_cfg.get("enabled", True):
            return []

        timeout = ddg_cfg.get("timeout", 5)
        try:
            logger.info(f"Querying DuckDuckGo Instant Answers for: {query}")
            url = "https://api.duckduckgo.com/"
            params = {
                "q": query,
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1"
            }
            headers = {"User-Agent": "Kruuboo-SLM/1.0"}
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code == 200:
                data = r.json()
                findings = []
                abstract = data.get("AbstractText", "").strip()
                heading = data.get("Heading", query)
                source_url = data.get("AbstractURL", "")

                if abstract:
                    findings.append({
                        "text": f"{heading}: {abstract}",
                        "source": f"DuckDuckGo ({data.get('AbstractSource', 'Web')})",
                        "url": source_url or f"https://duckduckgo.com/?q={requests.utils.quote(query)}"
                    })

                # Fallback to Definition or direct Answer if abstract is empty
                if not findings and data.get("Answer"):
                    findings.append({
                        "text": str(data["Answer"]).strip(),
                        "source": "DuckDuckGo Instant Answer",
                        "url": f"https://duckduckgo.com/?q={requests.utils.quote(query)}"
                    })
                elif not findings and data.get("Definition"):
                    findings.append({
                        "text": str(data["Definition"]).strip(),
                        "source": f"DuckDuckGo ({data.get('DefinitionSource', 'Definition')})",
                        "url": data.get("DefinitionURL") or f"https://duckduckgo.com/?q={requests.utils.quote(query)}"
                    })

                # If still empty, check related topics
                if not findings and data.get("RelatedTopics"):
                    for topic_item in data["RelatedTopics"][:2]:
                        if isinstance(topic_item, dict) and "Text" in topic_item and topic_item.get("Text"):
                            findings.append({
                                "text": topic_item["Text"].strip(),
                                "source": "DuckDuckGo Related Topic",
                                "url": topic_item.get("FirstURL", "")
                            })

                if findings:
                    return findings
        except Exception as e:
            logger.debug(f"DuckDuckGo query failed: {e}")
        return []

    def research_wikipedia(self, query: str, max_results: int = 3) -> List[Dict]:
        """Search Wikipedia encyclopedic articles for summaries."""
        wiki_cfg = self.web_cfg.get("wikipedia", {})
        if not wiki_cfg.get("enabled", True):
            return []

        limit = max_results or wiki_cfg.get("max_results", 3)
        try:
            logger.info(f"Searching Wikipedia for: {query}")
            search_results = wikipedia.search(query, results=limit)
            if not search_results:
                return []

            findings = []
            for result in search_results:
                try:
                    page = wikipedia.page(result, auto_suggest=False)
                    summary = page.summary.strip()
                    if summary:
                        findings.append({
                            "text": summary,
                            "source": f"Wikipedia: {result}",
                            "url": page.url
                        })
                except (wikipedia.DisambiguationError, wikipedia.PageError):
                    continue

            return findings
        except Exception as e:
            logger.debug(f"Wikipedia research failed for {query}: {e}")
            return []

    def research_arxiv(self, query: str, max_results: int = 2) -> List[Dict]:
        """Search ArXiv scientific and machine learning research paper repository."""
        arxiv_cfg = self.web_cfg.get("arxiv", {})
        if not arxiv_cfg.get("enabled", True):
            return []

        limit = max_results or arxiv_cfg.get("max_results", 2)
        timeout = arxiv_cfg.get("timeout", 5)
        try:
            logger.info(f"Searching ArXiv repository for: {query}")
            url = "http://export.arxiv.org/api/query"
            params = {
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": limit
            }
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200 and r.content:
                root = ET.fromstring(r.content)
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                findings = []
                for entry in root.findall("atom:entry", ns):
                    title_elem = entry.find("atom:title", ns)
                    summary_elem = entry.find("atom:summary", ns)
                    id_elem = entry.find("atom:id", ns)

                    if title_elem is not None and summary_elem is not None:
                        title = title_elem.text.strip().replace("\n", " ")
                        summary = summary_elem.text.strip().replace("\n", " ")
                        link = id_elem.text.strip() if id_elem is not None else ""
                        findings.append({
                            "text": f"Paper Title: {title}\nAbstract: {summary}",
                            "source": f"ArXiv ({title})",
                            "url": link
                        })
                return findings
        except Exception as e:
            logger.debug(f"ArXiv query failed for {query}: {e}")
        return []

    def research_topic(self, topic: str, max_results: Optional[int] = None) -> List[Dict]:
        """
        Search for a topic across the configured multi-provider fallback chain.
        Providers are executed sequentially until findings are successfully retrieved.
        """
        if not self.web_cfg.get("enabled", True):
            logger.debug("Web research disabled in config.yaml")
            return []

        cleaned_topic = clean_query(topic)
        logger.info(f"Cleaned research topic: '{topic}' -> '{cleaned_topic}'")

        # Provider fallback chain from config (defaults to standard robust sequence)
        provider_chain = self.web_cfg.get(
            "providers",
            ["wolfram", "duckduckgo", "wikipedia", "arxiv"]
        )

        math_keywords = [
            "+", "-", "*", "/", "=", "calculate", "solve", "math",
            "integrate", "derivative", "equation", "distance to",
            "speed of", "how far", "how many", "how much"
        ]
        is_math = any(kw in topic.lower() for kw in math_keywords)

        findings: List[Dict] = []

        # Provider execution map
        provider_handlers = {
            "wolfram": lambda q: self.research_wolfram(q) if is_math else [],
            "duckduckgo": lambda q: self.research_duckduckgo(q),
            "wikipedia": lambda q: self.research_wikipedia(q, max_results=max_results or 3),
            "arxiv": lambda q: self.research_arxiv(q, max_results=max_results or 2),
        }

        for provider in provider_chain:
            provider_name = str(provider).lower().strip()
            handler = provider_handlers.get(provider_name)
            if not handler:
                logger.debug(f"Unknown web research provider '{provider_name}', skipping.")
                continue

            try:
                results = handler(cleaned_topic)
                if results and len(results) > 0:
                    logger.info(f"Web research successfully retrieved {len(results)} items via [{provider_name}]")
                    findings.extend(results)
                    # Return if we have acquired enough grounded context
                    if len(findings) >= (max_results or 3):
                        return findings[:max_results or 3]
                else:
                    logger.debug(f"Provider [{provider_name}] returned 0 results for '{cleaned_topic}', falling back to next provider.")
            except Exception as e:
                logger.debug(f"Provider [{provider_name}] error: {e}, falling back.")

        if findings:
            return findings

        logger.info(f"All web research fallback providers exhausted for topic: '{cleaned_topic}'")
        return []

    def get_quick_fact(self, query: str) -> str:
        """Get a single quick fact from DuckDuckGo or Wikipedia."""
        ddg_facts = self.research_duckduckgo(query)
        if ddg_facts:
            return ddg_facts[0]["text"]

        try:
            return wikipedia.summary(query, sentences=2)
        except Exception:
            return ""


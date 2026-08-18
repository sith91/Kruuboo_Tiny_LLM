import time
import hashlib
from typing import Dict
from dataclasses import dataclass, field

@dataclass
class Document:
    """Document representation"""
    text: str
    source: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)
    
    def get_id(self) -> str:
        return hashlib.md5(f"{self.text[:100]}{self.source}".encode()).hexdigest()[:16]

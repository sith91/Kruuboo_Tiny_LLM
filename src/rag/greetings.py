import re
from typing import Dict, List, Optional

GREETINGS_MAP = {
    "hi": "Hi there! How can I help you today?",
    "hello": "Hello! How can I assist you today?",
    "hey": "Hey! How can I help you?",
    "greetings": "Greetings! How can I assist you today?",
    "thanks": "You're very welcome! Let me know if you need anything else.",
    "thank you": "You're very welcome! I'm happy to help.",
    "bye": "Goodbye! Have a great day!",
    "goodbye": "Goodbye! Take care and see you next time!",
    "how are you": "I'm doing great, thank you! How can I assist you today?",
    "how are you doing": "I'm doing well! How can I help you today?",
    "hows it going": "It's going great! What can I do for you today?",
    "how is it going": "Everything is going well! How can I help you today?",
    "how are you today": "I'm doing great, thank you! How are you doing today?",
    "how is your day": "My day is going great, thank you for asking! How is yours?",
    "how is everything": "Everything is running smoothly! How can I assist you?",
    "hows everything": "Everything is great! What are you working on today?",
    "good morning": "Good morning! Hope you have a wonderful day. How can I help you?",
    "good afternoon": "Good afternoon! How can I assist you today?",
    "good evening": "Good evening! How can I help you tonight?",
    "good night": "Good night! Sleep well and have pleasant dreams.",
    "nice to meet you": "Nice to meet you too! How can I help you today?",
    "pleased to meet you": "Pleased to meet you as well! What can I assist you with?",
    "what can you do": "I am Kruuboo Tiny LLM (105.4M parameters). I can chat conversationally and answer questions across a wide range of topics using Native 768d RAG and rStar reasoning!",
    "who are you": "I am Kruuboo Tiny LLM, a self-contained conversational AI assistant running locally.",
    "what are you": "I am a local Small Language Model (SLM) with native semantic retrieval and rStar reasoning capabilities.",
    "i feel very lonely": "I'm really sorry you're feeling lonely. I'm right here with you! What's on your mind? We can chat about anything you like.",
    "i feel very lonley": "I'm really sorry you're feeling lonely. I'm right here with you! What's on your mind? We can chat about anything you like.",
    "i feel lonely": "I'm here for you! You're never truly alone. Would you like to talk about what's going on, or just chat about something fun?",
    "i feel lonley": "I'm here for you! You're never truly alone. Would you like to talk about what's going on, or just chat about something fun?",
    "i am lonely": "I'm here with you! Feel free to share what's on your mind, or we can talk about movies, books, or any topic you enjoy.",
    "i am lonley": "I'm here with you! Feel free to share what's on your mind, or we can talk about movies, books, or any topic you enjoy.",
    "i'm lonely": "I'm right here! Let's talk—what kind of things make you feel relaxed or happy?",
    "i'm lonley": "I'm right here! Let's talk—what kind of things make you feel relaxed or happy?",
    "i feel sad": "I'm so sorry you're feeling down. I'm here to listen and keep you company. What happened?",
    "i'm sad": "I'm here for you. Take a deep breath. Is there anything specific on your mind, or would you like a comforting chat?",
    "i need a friend": "I'm always happy to be here as your friend! What's on your mind today?",
    "can we talk": "Of course! I'm all ears. What would you like to talk about?",
    "are you there": "Yes, I'm right here! How can I help you?"
}

NON_NAME_WORDS = {
    # Articles, determiners, prepositions, conjunctions
    "a", "an", "the", "this", "that", "these", "those", "here", "there",
    "in", "on", "at", "to", "for", "from", "with", "about", "by", "into",
    "through", "during", "before", "after", "above", "below", "and", "or",
    "but", "if", "because", "as", "until", "while", "of", "off", "out",
    "over", "under", "again", "further", "then", "once", "so", "than", "too",
    "very", "just", "also", "all", "any", "some", "every", "no", "not",

    # Pronouns & Auxiliaries
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us",
    "them", "my", "your", "his", "their", "our", "mine", "yours", "hers",
    "ours", "theirs", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "can", "could", "shall",
    "should", "will", "would", "may", "might", "must",

    # Common states, emotions, feelings, and adjectives
    "tired", "exhausted", "sleepy", "curious", "interested", "bored",
    "hungry", "thirsty", "full", "sad", "unhappy", "depressed", "lonely",
    "lonley", "happy", "glad", "joyful", "excited", "good", "fine", "great",
    "well", "okay", "ok", "bad", "sick", "ill", "angry", "mad", "furious",
    "confused", "lost", "stuck", "busy", "free", "ready", "done", "finished",
    "sure", "unsure", "afraid", "scared", "worried", "anxious", "nervous",
    "calm", "relaxed", "stressed", "alive", "dead", "hurt", "new", "old",
    "young", "human", "bot", "robot", "ai", "real", "fake", "sorry",
    "back", "here", "there", "home", "away", "alone",

    # Common verbs & gerunds
    "asking", "wondering", "looking", "trying", "working", "learning",
    "building", "coding", "testing", "searching", "thinking", "hoping",
    "writing", "reading", "playing", "living", "staying", "going", "coming",
    "waiting", "studying", "feeling", "doing", "getting", "making", "having",
    "using", "giving", "taking", "seeing", "saying", "telling", "talking",
    "eating", "drinking", "sleeping", "running", "walking", "driving",
    "flying", "sitting", "standing", "helping", "seeking", "exploring",

    # Common roles & professions (when stated as 'I am a ...')
    "student", "developer", "programmer", "engineer", "scientist", "doctor",
    "nurse", "teacher", "professor", "writer", "author", "designer", "artist",
    "musician", "manager", "director", "user", "guest", "admin", "tester",
    "researcher", "intern", "coder", "beginner", "expert", "pro"
}


def is_valid_name(name: str) -> bool:
    """Validate that candidate string is a plausible user name and not an adjective, gerund, or state."""
    if not name:
        return False

    name_clean = name.strip("?.!, \n\t").lower()
    if len(name_clean) < 2 or len(name_clean) > 30:
        return False

    tokens = name_clean.split()
    if len(tokens) > 3:
        return False

    for tok in tokens:
        if tok in NON_NAME_WORDS:
            return False
        # Reject gerunds (-ing) or past-participle adjectives (-ed)
        if (tok.endswith("ing") or tok.endswith("ed")) and len(tok) > 4:
            return False
        # Reject adverbs (-ly)
        if tok.endswith("ly") and len(tok) > 4:
            return False
        # Reject numeric values
        if tok.isdigit():
            return False

    return True


def extract_user_name_from_history(history: Optional[List[Dict]]) -> str:
    """Scan history to locate the user's name if they have explicitly introduced themselves."""
    if not history:
        return "my friend"

    user_patterns = [
        r"(?:my name is|call me)\s+([a-zA-Z0-9_\-]+)",
        r"(?:^|\b)(?:i am|i'm)\s+([a-zA-Z0-9_\-]+)",
    ]

    for turn in reversed(history):
        if turn.get("role") == "user":
            content = turn.get("content", "").strip()
            # If the user message is a long complex sentence, skip 'i am' extraction
            for pat in user_patterns:
                match = re.search(pat, content, re.IGNORECASE)
                if match:
                    candidate = match.group(1).strip("?.!, \n\t")
                    if is_valid_name(candidate):
                        # Additional check: If matched via 'i am' / 'i'm', ensure short statement
                        if "my name is" in content.lower() or "call me" in content.lower() or len(content.split()) <= 4:
                            return candidate.title()

    return "my friend"


def check_user_intro(clean_q: str) -> Optional[str]:
    """Check if the user is introducing themselves (e.g. 'My name is Sithija', 'Call me Sithija', 'I am Sithija')."""
    text = clean_q.strip()

    # 1. High-confidence explicit intro patterns
    explicit_patterns = [
        r"^my name is\s+([a-zA-Z0-9_\-\s]+)",
        r"^call me\s+([a-zA-Z0-9_\-]+)",
    ]
    for pat in explicit_patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            candidate = match.group(1).split()[0].strip("?.!, \n\t")
            if is_valid_name(candidate):
                return candidate.title()

    # 2. Ambiguous patterns ('i am X', "i'm X", 'this is X') - strictly guarded
    # Reject multi-word statements, questions, or sentences with verbs/prepositions following "I am"
    words = text.split()
    if len(words) <= 4:
        ambiguous_patterns = [
            r"^i am\s+([a-zA-Z0-9_\-]+)$",
            r"^i'm\s+([a-zA-Z0-9_\-]+)$",
            r"^this is\s+([a-zA-Z0-9_\-]+)$",
        ]
        for pat in ambiguous_patterns:
            match = re.search(pat, text, re.IGNORECASE)
            if match:
                candidate = match.group(1).strip("?.!, \n\t")
                if is_valid_name(candidate):
                    return candidate.title()

    return None


def check_naming_request(clean_q: str) -> Optional[str]:
    """Check if the user is assigning a name to the bot, returns the name if found."""
    bot_naming_patterns = [
        r"(?:i will name you as|your name is|call yourself|i'll name you|name you as)\s+([a-zA-Z0-9_\-\s]+)"
    ]
    for pattern in bot_naming_patterns:
        match = re.search(pattern, clean_q, re.IGNORECASE)
        if match:
            bot_name = match.group(1).strip("?.!, \n\t").title()
            if bot_name and is_valid_name(bot_name):
                return bot_name
    return None


def check_bot_name_query(clean_q: str) -> bool:
    """Check if the user is asking for the bot's name."""
    bot_patterns = [
        r"(?:what(?:'s| is)|tell me)?\s*your name",
        r"who are you",
        r"what do they call you",
        r"what should i call you",
        r"what is your identity",
    ]
    return any(re.search(pat, clean_q, re.IGNORECASE) for pat in bot_patterns)


def check_user_name_query(clean_q: str) -> bool:
    """Check if the user is asking for their own name."""
    user_name_patterns = [
        r"(?:what(?:'s| is)|tell me)?\s*my name",
        r"who am i",
        r"do you (?:know|remember)\s+(?:what\s+)?my name",
        r"remember my name",
        r"know my name",
        r"what do you call me",
    ]
    return any(re.search(pat, clean_q, re.IGNORECASE) for pat in user_name_patterns)



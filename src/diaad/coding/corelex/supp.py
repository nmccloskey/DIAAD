
urls = {
    "BrokenWindow": {
        "accuracy": "https://docs.google.com/spreadsheets/d/12SAkAG8VCAkhCFv4ceJiqgRZ7U9-P9bEcet--hDeW2s/export?format=csv&gid=1059193656",
        "efficiency": "https://docs.google.com/spreadsheets/d/12SAkAG8VCAkhCFv4ceJiqgRZ7U9-P9bEcet--hDeW2s/export?format=csv&gid=1542250565"
    },
    "RefusedUmbrella": {
        "accuracy": "https://docs.google.com/spreadsheets/d/1oYiwnUdO0dOsFVTmdZBCxkAQc5Ui-71GhUSchK_YY44/export?format=csv&gid=1670315041",
        "efficiency": "https://docs.google.com/spreadsheets/d/1oYiwnUdO0dOsFVTmdZBCxkAQc5Ui-71GhUSchK_YY44/export?format=csv&gid=1362214973"
    },
    "CatRescue": {
        "accuracy": "https://docs.google.com/spreadsheets/d/1sTvSX0Ws0kPTw-5HHyY8JO2CubqWVgEzDvE5BuGSefc/export?format=csv&gid=1916867784",
        "efficiency": "https://docs.google.com/spreadsheets/d/1sTvSX0Ws0kPTw-5HHyY8JO2CubqWVgEzDvE5BuGSefc/export?format=csv&gid=1346760459"
    },
    "Cinderella": {
        "accuracy": "https://docs.google.com/spreadsheets/d/1fpDq7aTrKVkfjdv8ka7BS5_iHEJ8HHI-q9nJI6wDAEA/export?format=csv&gid=280451139",
        "efficiency": "https://docs.google.com/spreadsheets/d/1fpDq7aTrKVkfjdv8ka7BS5_iHEJ8HHI-q9nJI6wDAEA/export?format=csv&gid=285651009"
    },
    "Sandwich": {
        "accuracy": "https://docs.google.com/spreadsheets/d/1o29bBQbyNlmtL05kkTuLV6z5auz1msDeLSxIO1p_3EA/export?format=csv&gid=342443913",
        "efficiency": "https://docs.google.com/spreadsheets/d/1o29bBQbyNlmtL05kkTuLV6z5auz1msDeLSxIO1p_3EA/export?format=csv&gid=2140143611"
    }
}


# Define tokens for each scene
scene_tokens = {
    'BrokenWindow': [
        "a", "and", "ball", "be", "boy", "break", "go", "he", "in", "it", 
        "kick", "lamp", "look", "of", "out", "over", "play", "sit", "soccer", 
        "the", "through", "to", "up", "window"
    ],
    'CatRescue': [
        "a", "and", "bark", "be", "call", "cat", "climb", "come",
        "department", "dog", "down", "father", "fire", "fireman", "get",
        "girl", "go", "have", "he", "in", "ladder", "little", "not", "out",
        "she", "so", "stick", "the", "their", "there", "to", "tree", "up", "with"
    ],
    'RefusedUmbrella': [
        "a", "and", "back", "be", "boy", "do", "get", "go", "have", "he", "home",
        "i", "in", "it", "little", "mother", "need", "not", "out", "rain",
        "say", "school", "she", "so", "start", "take", "that", "the", "then",
        "to", "umbrella", "walk", "wet", "with", "you"
    ],
    'Cinderella': [
        "a", "after", "all", "and", "as", "at", "away", "back", "ball", "be",
        "beautiful", "because", "but", "by", "cinderella", "clock", "come", "could",
        "dance", "daughter", "do", "dress", "ever", "fairy", "father", "find", "fit",
        "foot", "for", "get", "girl", "glass", "go", "godmother", "happy", "have",
        "he", "home", "horse", "house", "i", "in", "into", "it", "know", "leave",
        "like", "little", "live", "look", "lose", "make", "marry", "midnight",
        "mother", "mouse", "not", "of", "off", "on", "one", "out", "prince",
        "pumpkin", "run", "say", "'s", "she", "shoe", "sister", "slipper", "so", "strike",
        "take", "tell", "that", "the", "then", "there", "they", "this", "time",
        "to", "try", "turn", "two", "up", "very", "want", "well", "when", "who",
        "will", "with"
    ],
    'Sandwich': [
        "a", "and", "bread", "butter", "get", "it", "jelly", "knife", "of", "on",
        "one", "other", "out", "peanut", "piece", "put", "slice", "spread", "take",
        "the", "then", "to", "together", "two", "you"
    ]
}


lemma_dict = {
    # Pronouns and reflexives
    "its": "it", "itself": "it",
    "your": "you", "yours": "you", "yourself": "you",
    "him": "he", "himself": "he", "his": "he",
    "her": "she", "herself": "she",
    "them": "they", "themselves": "they", "their": "they", "theirs": "they",
    "me": "i", "my": "i", "mine": "i", "myself": "i",

    # Forms of "be"
    "is": "be", "are": "be", "was": "be", "were": "be", "am": "be",
    "being": "be", "been": "be", "bein": "be",

    # Parental variations
    "daddy": "father", "dad": "father", "papa": "father", "pa": "father",
    "mommy": "mother", "mom": "mother", "mama": "mother", "ma": "mother",

    # "-in" participles (casual speech)
    "breakin": "break", "goin": "go", "kickin": "kick", "lookin": "look",
    "playin": "play", "barkin": "bark", "callin": "call", "climbin": "climb",
    "comin": "come", "gettin": "get", "havin": "have", "stickin": "stick",
    "doin": "do", "needin": "need", "rainin": "rain", "sayin": "say",
    "startin": "start", "takin": "take", "walkin": "walk",

    # Additional verb forms and common variants
    "goes": "go", "gone": "go", "went": "go", "going": "go",
    "gets": "get", "got": "get", "getting": "get",
    "says": "say", "said": "say", "saying": "say",
    "takes": "take", "took": "take", "taking": "take",
    "looks": "look", "looked": "look", "looking": "look",
    "starts": "start", "started": "start", "starting": "start",
    "plays": "play", "played": "play", "playing": "play",

    # Noun variants
    "boys": "boy", "girls": "girl", "shoes": "shoe", "sisters": "sister",
    "trees": "tree", "windows": "window", "cats": "cat", "dogs": "dog",
    "pieces": "piece", "slices": "slice", "sandwiches": "sandwich",
    "fires": "fire", "ladders": "ladder", "balls": "ball",

    # Misc fix-ups
    "wanna": "want", "gonna": "go", "gotta": "get",
    "yall": "you", "aint": "not", "cannot": "could",

    # Additional verb forms
    "wants": "want", "wanted": "want", "wanting": "want",
    "finds": "find", "found": "find", "finding": "find",
    "makes": "make", "made": "make", "making": "make",
    "tries": "try", "tried": "try", "trying": "try",
    "tells": "tell", "told": "tell", "telling": "tell",
    "runs": "run", "ran": "run", "running": "run",
    "sits": "sit", "sat": "sit", "sitting": "sit",
    "knows": "know", "knew": "know", "knowing": "know",
    "walks": "walk", "walked": "walk", "walking": "walk",
    "leaves": "leave", "left": "leave", "leaving": "leave",
    "comes": "come", "came": "come", "coming": "come",
    "calls": "call", "called": "call", "calling": "call",
    "climbs": "climb", "climbed": "climb", "climbing": "climb",
    "breaks": "break", "broke": "break", "breaking": "break",
    "starts": "start", "started": "start", "starting": "start",
    "turns": "turn", "turned": "turn", "turning": "turn",
    "puts": "put", "putting": "put",  # 'put' is same for present/past

    # Copula contractions (useful if splitting fails elsewhere)
    "'m": "be", "'re": "be",  # context-dependent, but may help

    # More noun plurals
    "slippers": "slipper", "daughters": "daughter", "sons": "son",
    "knives": "knife", "pieces": "piece", "sticks": "stick",

    # Pronoun common errors
    "themself": "they", "our": "we", "ours": "we", "ourselves": "we", "we're": "we",

    # Additional contractions and speech forms
    "didnt": "did", "couldnt": "could", "wouldnt": "would", "shouldnt": "should",
    "wasnt": "was", "werent": "were", "isnt": "is", "aint": "not", "havent": "have",
    "hasnt": "have", "hadnt": "have", "dont": "do", "doesnt": "do", "didnt": "do",
    "did": "do", "does": "do", "doing": "do",

    # Articles
    "da": "the", "an": "a",

    # Spoken reductions
    "lemme": "let", "gimme": "give", "cmon": "come", "outta": "out",
    "inna": "in", "coulda": "could", "shoulda": "should", "woulda": "would",
}

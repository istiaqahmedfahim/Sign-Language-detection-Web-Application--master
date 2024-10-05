import keyboard

def phonetic_transform_word(word, mapping, diacritics_mapping):
    result = ""
    buffer = ""
    i = 0

    while i < len(word):
        char = word[i]

        buffer += char  # Add character to buffer

        # Check for three-letter diacritic combinations first (e.g., "khi")
        if i + 2 < len(word) and (buffer + word[i + 1] + word[i + 2]) in diacritics_mapping:
            buffer += word[i + 1] + word[i + 2]  # Add the next two characters to buffer
            result += diacritics_mapping[buffer]  # Add the diacritic form for "khi"
            buffer = ""  # Clear buffer after successful match
            i += 2  # Skip the next two characters since they are already processed
        # If not found, check for two-letter diacritic combinations (e.g., "ka")
        elif i + 1 < len(word) and (buffer + word[i + 1]) in diacritics_mapping:
            buffer += word[i + 1]  # Add the next vowel character to buffer
            result += diacritics_mapping[buffer]  # Add the diacritic form for "ka"
            buffer = ""  # Clear buffer after successful match
            i += 1  # Skip the next character since it's already processed
        else:
            # Check if buffer matches a single consonant or vowel
            if buffer in mapping:
                result += mapping[buffer]
                buffer = ""  # Clear buffer after match
            else:
                # Handle multi-character combinations (like 'kh', 'aa', etc.)
                if i + 1 < len(word) and (buffer + word[i + 1]) in mapping:
                    buffer += word[i + 1]
                    result += mapping[buffer]  # Add the mapped value
                    buffer = ""  # Clear the buffer
                    i += 1  # Skip the next character
                elif buffer[:-1] in mapping:
                    result += mapping[buffer[:-1]]
                    buffer = buffer[-1]  # Keep the last character in buffer

        i += 1

    # Add any remaining buffer (if any)
    if buffer in mapping:
        result += mapping[buffer]

    return result

# Phonetic mapping dictionary (for demonstration)
phonetic_mapping = {
    "a": "অ", "i": "ই", "ii": "ঈ", "u": "উ", "uu": "ঊ",
    "e": "এ", "oi": "ঐ", "o": "ও", "ou": "ঔ",
    "k": "ক", "kh": "খ", "g": "গ", "gh": "ঘ", "ng": "ঙ",
    "c": "চ", "ch": "ছ", "j": "জ", "jh": "ঝ", "ny": "ঞ",
    "t": "ট", "th": "ঠ", "d": "ড", "dh": "ঢ", "n": "ন",
    "p": "প", "ph": "ফ", "b": "ব", "bh": "ভ", "m": "ম",
    "r": "র", "l": "ল", "sh": "শ", "ss": "ষ", "s": "স", "h": "হ",
    # More consonants and mappings can be added here...
}

# Diacritics mapping for consonants followed by vowels (including three-letter cases)
diacritics_mapping = {
    "ma": "মা", "ki": "কি", "kii": "কী", "ku": "কু", "kuu": "কূ","aa": "আ",
    "ke": "কে", "kai": "কৈ", "ko": "কো", "kou": "কৌ",
    "kha": "খা", "khi": "খি", "khii": "খী", "khu": "খু", "khuu": "খূ","tha":"থা","shu": "শু","nu": "নু"
    # Add diacritic mappings for other consonants...
}

# Real-time word input detection
print("Start typing. Press 'Space' to transform each word. Type 'exit' to stop.")

current_word = ""  # To store the current word

def process_space():
    global current_word
    if current_word:
        # Transform the word and print the result
        transformed_word = phonetic_transform_word(current_word, phonetic_mapping, diacritics_mapping)
        print(f"Transformed: {transformed_word}")
        current_word = ""  # Clear word after transforming

def process_input(event):
    global current_word
    if event.name == "space":
        process_space()
    elif event.name == "backspace":
        current_word = current_word[:-1]  # Remove last character on backspace
    elif event.name == "enter":
        process_space()
        print("Exiting.")
        keyboard.unhook_all()  # Stop listening
    else:
        current_word += event.name  # Append new character to the current word

# Hook to listen to all keypresses
keyboard.on_press(process_input)

# Keep the program running
keyboard.wait('enter')
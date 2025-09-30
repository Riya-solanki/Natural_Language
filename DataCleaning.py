from BPETokenizer import BPETokenizer
import json
import os
import regex as re

# Directory containing the raw files
directory_path = '/content/Natural_Language/Final Corpus'

# Output directory (inside Natural_Language, parallel to Final Corpus)
output_dir = '/content/Natural_Language/Cleaned Corpus'
os.makedirs(output_dir, exist_ok=True)

def replace_short_text_between_double_dandas(text):
    pattern = r'।।.*?।।'
    def replace_match(match):
        matched_text = match.group(0)
        if len(matched_text) < 22:
            return '।।'
        else:
            return match.group(0)
    return re.sub(pattern, replace_match, text)

def replace_short_text_between_single_dandas(text):
    pattern = r'।.*?।'
    def replace_match(match):
        matched_text = match.group(0)
        if (len(matched_text) > 2) and (len(matched_text) < 22):
            return '।'
        else:
            return match.group(0)
    return re.sub(pattern, replace_match, text)

Indx = 1

for filename in os.listdir(directory_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(directory_path, filename)
        
        print(Indx)
        Indx += 1

        with open(file_path, "r", encoding="utf-8") as file:
            Text = file.read()   
            CleanedText = '\n'.join([line for line in Text.splitlines()])
            del Text

            # Cleaning steps
            CleanedText = CleanedText.replace('.', '').replace('*', '').replace('|', '।')
            CleanedText = re.sub(r'%%.*?\n', '', CleanedText)
            CleanedText = re.sub(r'<.*?>', '', CleanedText)
            CleanedText = re.sub(r' +', ' ', CleanedText)
            CleanedText = re.sub(r'<BR>', '', CleanedText)
            CleanedText = re.sub(r'\n +', '\n', CleanedText)
            CleanedText = re.sub(r'\(\(.*?\)\)', ' ', CleanedText)
            CleanedText = re.sub(r'\(.*?\)', ' ', CleanedText)
            CleanedText = re.sub(r' [%$&\\#]', '', CleanedText)
            CleanedText = re.sub(r'[%$&\\#]', '', CleanedText)
            CleanedText = re.sub(r'(?<=\d)-(?=\d)', '', CleanedText)
            CleanedText = re.sub(r'[0-9][a-z]', '', CleanedText)
            CleanedText = re.sub(r'[0-9],', '', CleanedText)
            CleanedText = re.sub(r'}', '', CleanedText)
            CleanedText = re.sub(r" ''", ' ā', CleanedText)
            CleanedText = re.sub(r" '", ' a', CleanedText)
            CleanedText = re.sub(r"\n''", '\nā', CleanedText)
            CleanedText = re.sub(r"\n'", '\na', CleanedText)
            CleanedText = re.sub(r'[0-9\.]', '', CleanedText)
            CleanedText = re.sub(r' +', ' ', CleanedText)
            CleanedText = re.sub(r'\[\d+\]', '', CleanedText)
            CleanedText = re.sub(r'\[\]', '', CleanedText)
            CleanedText = re.sub(r'/', '।', CleanedText)

            CleanedText = replace_short_text_between_double_dandas(CleanedText)
            CleanedText = replace_short_text_between_single_dandas(CleanedText)
            CleanedText = re.sub(r'।{3,}', '।।', CleanedText)
            CleanedText = re.sub(r'\s,\s', ', ', CleanedText)
            CleanedText = re.sub(r'\s;\s', '; ', CleanedText)
            CleanedText = re.sub(r':', '', CleanedText)

            CleanedText = re.sub(r'\n{3,}', '\n\n', CleanedText)
            CleanedText = re.sub(r'(\r\n){3,}', '\n\n', CleanedText)
            CleanedText = re.sub(r' +', ' ', CleanedText)
            CleanedText = re.sub(r'\n +', '\n', CleanedText)
            CleanedText = re.sub(r'\n{3,}', '\n\n', CleanedText)
            CleanedText = re.sub(r'(\r\n){3,}', '\n\n', CleanedText)

            CleanedText = re.sub(r'\{\d+\}', '', CleanedText)
            CleanedText = re.sub(r'\{\}', '', CleanedText)
            CleanedText = re.sub(r'\n__+', '\n\n', CleanedText)
            CleanedText = re.sub(r'\n--+', '\n\n', CleanedText)
            CleanedText = re.sub(r'\n==+', '\n\n', CleanedText)
            CleanedText = re.sub(r'--+', '', CleanedText)
            CleanedText = re.sub(r'=', '', CleanedText)
            CleanedText = re.sub(r'_', '', CleanedText)
            CleanedText = re.sub(r'\+', ' ', CleanedText)
            CleanedText = re.sub(r' +', ' ', CleanedText)
            CleanedText = re.sub(r'।। ।।', '।।', CleanedText)

        # Avoid duplicate -Cleaned suffix
        base_name = filename.replace('-Cleaned', '')  
        output_file = os.path.join(output_dir, base_name[:-4] + '-Cleaned.txt')

        # Write the cleaned content back to the file
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(CleanedText)

print("✅ All files have been cleaned, processed, and saved under /content/Natural_Language/Cleaned Corpus/")
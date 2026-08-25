import fitz
import sys

doc = fitz.open(r'c:\Users\burra\F1-Simulation\2512.00640v1_highlighted.pdf')
text = '\n'.join([page.get_text() for page in doc])
# Write to a text file since stdout may have issues
with open(r'c:\Users\burra\F1-Simulation\paper_text.txt', 'w', encoding='utf-8') as f:
    f.write(text)
print(f"Extracted {len(text)} characters to paper_text.txt")

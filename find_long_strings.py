#!/usr/bin/env python3
import re
import os

def find_long_strings(file_path, min_length=40):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all string literals
    # This regex finds strings in single or double quotes
    pattern = r'["\']([^"\'\\]*(\\.[^"\'\\]*)*)["\']'
    matches = re.findall(pattern, content)
    
    long_strings = []
    for match in matches:
        # Get the actual string content (first group)
        string_content = match[0] if isinstance(match, tuple) else match
        if (len(string_content) >= min_length and 
            not string_content.startswith('http') and 
            not string_content.startswith('file:') and
            not string_content.startswith('C:') and
            not string_content.startswith('/') and
            '\\' not in string_content[:10]):  # Skip file paths
            long_strings.append(string_content)
    
    return long_strings

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    app_file = os.path.join(script_dir, 'secourses_app.py')
    
    long_strings = find_long_strings(app_file)
    unique_strings = sorted(set(long_strings))
    
    print(f'Found {len(unique_strings)} unique strings 40+ characters in secourses_app.py')
    print('\nFirst 30 strings:')
    for i, s in enumerate(unique_strings[:30]):
        if len(s) > 80:
            print(f'{i+1:2d}. {s[:77]}...')
        else:
            print(f'{i+1:2d}. {s}')
    
    if len(unique_strings) > 30:
        print(f'\n... and {len(unique_strings) - 30} more strings') 
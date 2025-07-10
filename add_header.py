# *******************************************************
# * FILE: add_header
# * AUTHOR: Pedro Encarnação
# * DATE: 09/07/2025
# * LICENSE: "CC BY-NC-SA 4.0"
# *******************************************************

import os
from datetime import datetime

# Customize your header here
AUTHOR = "Pedro Encarnação"
LICENSE = "CC BY-NC-SA 4.0"

HEADER_TEMPLATE = """# *******************************************************
# * FILE: {filename}
# * AUTHOR: {author}
# * DATE: {date}
# * LICENSE: {license}
# *******************************************************

"""

def has_header(content):
    return "LICENSE: " in content and "AUTHOR: " in content

def prepend_header(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    if has_header(content):
        print(f"Skipped (header exists): {file_path}")
        return

    filename = os.path.basename(file_path)
    date_str = datetime.now().strftime("%Y-%m-%d")
    header = HEADER_TEMPLATE.format(filename=filename, author=AUTHOR, date=date_str, license=LICENSE)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(header + content)

    print(f"Updated: {file_path}")

def process_directory(root_dir):
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                prepend_header(file_path)

# 🔁 Set your project root here
if __name__ == "__main__":
    project_root = "."  # or replace with your absolute path
    process_directory(project_root)
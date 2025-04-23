#  Copyright (c) 2025. Pedro Encarnação . Universidade de Aveiro LICENSE: CC BY-NC-SA 4.0 # ****************************
#

# *******************************************************
# * FILE: automatically_issue_creation
# * AUTHOR: Pedro Encarnação
# * DATE: 23/01/2025
# * LICENSE: "CC BY-NC-SA 4.0"
# *******************************************************

"""
This creation automatically creates issues in a GitHub repository for documentation files.
"""

import os
import requests

# Configuration
GITHUB_TOKEN = "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"
REPO_OWNER = "crispim-enc"
REPO_NAME = "Infinity-Tomographic-Reconstruction"
API_URL = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/issues"

# Headers for GitHub API
HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

# Function to create an issue
def create_issue(title, body):
    data = {
        "title": title,
        "body": body
    }
    response = requests.post(API_URL, headers=HEADERS, json=data)
    if response.status_code == 201:
        print(f"Issue created: {title}")
    else:
        print(f"Failed to create issue: {response.status_code} - {response.json()}")

# Function to scan for documentation files
def scan_and_create_issues():
    for root, _, files in os.walk("."):  # Scans the current directory
        for file in files:
            if file.endswith((".md", ".rst", ".txt")):  # Adjust extensions as needed
                file_path = os.path.join(root, file)
                title = f"Review Documentation: {file}"
                body = f"Please review the documentation in the file `{file_path}`."
                create_issue(title, body)

# Run the script
if __name__ == "__main__":
    scan_and_create_issues()

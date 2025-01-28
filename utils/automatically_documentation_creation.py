#  Copyright (c) 2025. Pedro Encarnação . Universidade de Aveiro LICENSE: CC BY-NC-SA 4.0 # ****************************
#

# *******************************************************
# * FILE: automatically_documentation_creation
# * AUTHOR: Pedro Encarnação
# * DATE: 23/01/2025
# * LICENSE: "CC BY-NC-SA 4.0"
# *******************************************************

import os
import requests
from datetime import datetime, timedelta

GITHUB_TOKEN = "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"
GITHUB_TOKEN ="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"
GITHUB_TOKEN ="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"
REPO_OWNER = "crispim-enc"
REPO_NAME = "Infinity-Tomographic-Reconstruction"
API_URL = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/issues"

# Configuration

PROJECT_ID = "1"  # Replace with your project's numeric ID
TODO_COLUMN_ID = "1"  # Replace with your project's ToDo column ID
ISSUES_PER_DAY = 8  # Limit to 8 issues per day

# GitHub API Endpoints
CREATE_ISSUE_URL = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/issues"
PROJECT_CARD_URL = f"https://api.github.com/projects/columns/{TODO_COLUMN_ID}/cards"


# Headers for GitHub API
HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

# Function to create an issue
def create_issue(title, body, labels, assignees):
    data = {
        "title": title,
        "body": body,
        "labels": labels,
        "assignees": assignees,
    }
    # check if issue already exists

    response = requests.get(API_URL, headers=HEADERS)
    # if response.status_code == 200:
    #     print(f"Issue already exists: {title}")
    #     return None, None

    if response.status_code == 422:
        print(f"Issue already exists: {title}")
        return None, None

    response = requests.post(CREATE_ISSUE_URL, headers=HEADERS, json=data)
    # do not create issue if it already exists

    if response.status_code == 201:
        issue_url = response.json()["url"]
        issue_number = response.json()["number"]
        print(f"Issue created: {title} (#{issue_number})")
        return issue_url, issue_number
    else:
        print(f"Failed to create issue: {response.status_code} - {response.json()}")
        return None, None

# Function to add an issue to a GitHub project
def add_to_project(issue_url):
    data = {"content_type": "Issue", "content_url": issue_url}
    response = requests.post(PROJECT_CARD_URL, headers=HEADERS, json=data)
    if response.status_code == 201:
        print("Issue added to project.")
    else:
        print(f"Failed to add issue to project: {response.status_code} - {response.json()}")

# Function to schedule issues
def schedule_issues(files):
    current_time = datetime.now()
    issue_time = current_time.replace(minute=0, second=0, microsecond=0)
    daily_count = 0

    for file in files:
        if daily_count == ISSUES_PER_DAY:
            # Move to the next day
            issue_time += timedelta(days=1)
            issue_time = issue_time.replace(hour=9)  # Start at 9 AM
            daily_count = 0

        # Create issue title and body
        title = f"Create Documentation: {file}"
        body = f"Documentation is required for the Python file `{file}`.\n\nPlease provide detailed and clear documentation for all functions, classes, and methods."
        labels = ["Documentation"]
        assignees = [REPO_OWNER]  # Assign to yourself

        # Create the issue
        issue_url, issue_number = create_issue(title, body, labels, assignees)

        if issue_url:
            # Add the issue to the project in the ToDo column
            add_to_project(issue_url)

            print(f"Issue #{issue_number} scheduled at {issue_time.strftime('%Y-%m-%d %H:%M')}")

        # Increment time for the next issue
        issue_time += timedelta(hours=1)
        daily_count += 1

# Function to scan for Python files
def find_python_files():
    python_files = []
    for root, _, files in os.walk("."):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))
    return python_files

# Main script
if __name__ == "__main__":
    python_files = find_python_files()
    if python_files:
        print(f"Found {len(python_files)} Python files.")
        schedule_issues(python_files)
    else:
        print("No Python files found.")
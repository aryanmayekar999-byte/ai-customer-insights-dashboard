def assign_severity(issue):
    issue = issue.lower()
    if "crash" in issue or "failure" in issue:
        return "High"
    elif "slow" in issue or "delay" in issue:
        return "Medium"
    else:
        return "Low"
import matplotlib.pyplot as plt

def plot_issues(issues):
    counts = [len(issue.split()) for issue in issues]

    plt.figure()
    plt.barh(issues, counts)
    plt.xlabel("Impact Score")
    plt.title("Issue Importance")

    return plt
import json

import json
import re

def safe_parse_json(text):
    try:
        return json.loads(text)
    except:
        try:
            # Extract JSON block using regex
            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            return None
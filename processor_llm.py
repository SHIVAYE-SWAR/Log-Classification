import ollama
import re

def classify_with_llm(log_msg):
    """
    Classify the log message into one of these categories:
    (1) Workflow Error, (2) Deprecation Warning.
    If not classifiable, return "Unclassified".
    """
    prompt = f'''Classify the log message into one of these categories: 
    (1) Workflow Error, (2) Deprecation Warning.
    If you can't figure out a category, use "Unclassified".
    Put the category inside <category> </category> tags.
    Log message: {log_msg}'''

    response = ollama.chat(
        model='gemma:2b',  # Change to your locally available model name
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    content = response['message']['content']
    match = re.search(r'<category>(.*?)<\/category>', content, flags=re.DOTALL)
    category = "Unclassified"
    if match:
        category = match.group(1).strip()

    return category


if __name__ == "__main__":
    print(classify_with_llm(
        "Case escalation for ticket ID 7324 failed because the assigned support agent is no longer active."))
    print(classify_with_llm(
        "The 'ReportGenerator' module will be retired in version 4.0. Please migrate to the 'AdvancedAnalyticsSuite' by Dec 2025"))
    print(classify_with_llm("System reboot initiated by user 12345."))

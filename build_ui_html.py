import json
import os

TEMPLATE_PATH = "template_analytics.html"
UI_DATA_PATH = "ui_data.json"
WORKSPACE_OUTPUT = "index.html"
ARTIFACT_OUTPUT = r"C:\Users\hp\.gemini\antigravity\brain\8935361b-8cbc-4832-8e9e-ee920f4494a0\index.html"

def main():
    with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
        template = f.read()

    with open(UI_DATA_PATH, "r", encoding="utf-8") as f:
        data_json = f.read()

    final_html = template.replace("__UI_DATA_JSON__", data_json)

    with open(WORKSPACE_OUTPUT, "w", encoding="utf-8") as f:
        f.write(final_html)
    print(f"Successfully generated: {os.path.abspath(WORKSPACE_OUTPUT)} ({len(final_html)} bytes)")

    os.makedirs(os.path.dirname(ARTIFACT_OUTPUT), exist_ok=True)
    with open(ARTIFACT_OUTPUT, "w", encoding="utf-8") as f:
        f.write(final_html)
    print(f"Successfully generated: {ARTIFACT_OUTPUT}")

if __name__ == "__main__":
    main()

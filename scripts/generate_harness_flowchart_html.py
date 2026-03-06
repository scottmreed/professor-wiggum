#!/usr/bin/env python3
"""
Generate HTML from Mermaid harness configuration flowchart
"""
from pathlib import Path


def create_html(mermaid_content, output_file, title="Mechanistic Agent Harness Configuration"):
    """Create HTML file with interactive Mermaid diagram."""

    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        mermaid.initialize({{
            startOnLoad: true,
            theme: 'default',
            securityLevel: 'loose',
            flowchart: {{
                useMaxWidth: true,
                htmlLabels: true,
                curve: 'basis'
            }}
        }});
    </script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 20px;
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            margin-bottom: 10px;
        }}
        .subtitle {{
            color: #7f8c8d;
            text-align: center;
            margin-bottom: 30px;
            font-size: 14px;
        }}
        .mermaid {{
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 800px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <p class="subtitle">Visual representation of the mechanistic agent harness configuration and execution flow</p>
        <div class="mermaid">
{mermaid_content}
        </div>
    </div>
</body>
</html>"""

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_template)
    print(f"HTML generated: {output_file}")


def main():
    # Get paths
    project_root = Path(__file__).parent.parent
    mermaid_file = project_root / "docs" / "diagrams" / "Harness_Configuration_Flowchart.mmd"
    output_file = project_root / "Harness_Configuration_Flowchart.html"

    if not mermaid_file.exists():
        print(f"Error: {mermaid_file} not found")
        return

    print(f"Reading mermaid diagram from {mermaid_file}...")
    with open(mermaid_file, 'r', encoding='utf-8') as f:
        mermaid_content = f.read()

    print(f"Generating HTML: {output_file}...")
    create_html(mermaid_content, output_file)
    print("Done!")


if __name__ == "__main__":
    main()
"""Run the weekly briefing and write it to a file."""

from datetime import datetime
from briefing import generate_briefing


if __name__ == "__main__":
    print("=" * 70)
    print("  Generating weekly gold market briefing")
    print("=" * 70)

    briefing = generate_briefing()

    # Write to a dated file
    today = datetime.now().strftime("%Y-%m-%d")
    filename = f"briefing-{today}.md"
    with open(filename, "w") as f:
        f.write(briefing)

    print(f"\n[Briefing written to {filename}]\n")
    print("=" * 70)
    print(briefing)
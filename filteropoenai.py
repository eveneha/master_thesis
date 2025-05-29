import json
from datetime import datetime

# Load JSON
with open("/media/eveneiha/UBUNTU 22_0/conversations.json", "r") as f:
    data = json.load(f)

def is_2025_or_later(msg):
    ts = msg.get("create_time")
    if ts:
        return datetime.fromtimestamp(float(ts)).year >= 2025
    return False

# Counters
simple_text_count = 0
complex_text_count = 0
image_gen_count = 0
file_upload_count = 0
skipped_count = 0

image_keywords = ["generate image", "dall·e", "draw", "create an image", "cartoon", "photo of"]
file_keywords = ["upload", "csv", "json", "file", "log", "analyze this", "parse", "attached"]

for convo in data:
    mapping = convo.get("mapping", {})
    for node in mapping.values():
        msg = node.get("message")
        if (
            msg
            and msg.get("author", {}).get("role") == "user"
            and is_2025_or_later(msg)
        ):
            content_field = msg.get("content", {})
            parts = content_field.get("parts")

            # Try to extract message text
            if isinstance(parts, list) and parts and isinstance(parts[0], str):
                content_text = parts[0]
            elif isinstance(parts, str):
                content_text = parts
            else:
                skipped_count += 1
                continue  # Skip malformed entry

            content_lower = content_text.lower()
            word_count = len(content_lower.split())

            # Classify
            if any(kw in content_lower for kw in image_keywords):
                image_gen_count += 1
            elif any(kw in content_lower for kw in file_keywords):
                file_upload_count += 1
            elif word_count > 100:
                complex_text_count += 1
            else:
                simple_text_count += 1

# Report
print("=== Usage Summary (2025 onward) ===")
print(f"Simple text prompts      : {simple_text_count}")
print(f"Complex text prompts     : {complex_text_count}")
print(f"Image generation prompts : {image_gen_count}")
print(f"File-related prompts     : {file_upload_count}")
print(f"Skipped (unreadable)     : {skipped_count}")

total = simple_text_count + complex_text_count + image_gen_count + file_upload_count
print(f"Total classified prompts : {total}")


from pathlib import Path
from lxml import etree

TRAIN_DIR = Path("data/raw/train")
TEST_DIR = Path("data/raw/test")


def inspect_folder(folder):
    files = list(folder.rglob("*.xml"))

    print(f"\nFolder: {folder}")
    print(f"XML files found: {len(files)}")

    return files


# Count files
train_files = inspect_folder(TRAIN_DIR)
test_files = inspect_folder(TEST_DIR)

if not train_files:
    raise FileNotFoundError("No XML files found in training folder.")

# Select first XML file
sample_file = train_files[0]

print("\n========== SAMPLE FILE ==========")
print(sample_file.name)

# Read XML content
xml_content = sample_file.read_text(
    encoding="utf-8",
    errors="ignore"
)

print("\n========== RAW XML PREVIEW ==========")
print(xml_content[:5000])

# Parse XML
try:
    root = etree.fromstring(xml_content.encode("utf-8"))

    print("\n========== XML STRUCTURE ==========")
    print("Root tag:", root.tag)

    print("\n========== XML ELEMENTS ==========")

    for element in root.iter():
        print(
            "Tag:", element.tag,
            "| Attributes:", element.attrib,
            "| Text:", (element.text or "").strip()[:100]
        )

except etree.XMLSyntaxError as error:
    print("XML parsing error:", error)

print("\nXML inspection completed!")

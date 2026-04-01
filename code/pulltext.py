from pathlib import Path

folder = Path(".")  # change this to your folder path if needed
output_file = folder / "all_python_files_contents.txt"

py_files = sorted(folder.glob("*.py"))

with output_file.open("w", encoding="utf-8") as out:
    for i, file_path in enumerate(py_files):
        content = file_path.read_text(encoding="utf-8", errors="replace")
        out.write(f"{file_path.name}\n")
        out.write(content)
        if i != len(py_files) - 1:
            out.write("\n\n\n")

print(f"Saved to: {output_file}")

"""
Lesson runner with proper UTF-8 encoding for Windows.
Usage: python run_lesson.py <lesson_number>
"""
import sys
import os

# Force UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

if len(sys.argv) < 2:
    print("Usage: python run_lesson.py <lesson_number>")
    print("Example: python run_lesson.py 1")
    sys.exit(1)

lesson_num = sys.argv[1]
lesson_file = f"lessons/lesson_{lesson_num.zfill(2)}_*.py"

# Find the lesson file
import glob
matches = glob.glob(lesson_file)

if not matches:
    print(f"Error: Lesson {lesson_num} not found")
    sys.exit(1)

# Run the lesson
lesson_path = matches[0]
print(f"Running: {lesson_path}\n")

with open(lesson_path, encoding='utf-8') as f:
    code = f.read()
    exec(code)

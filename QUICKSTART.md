# Quick Start Guide

## ✅ Setup Complete!

Your Bible corpus is ready:
- **9 translations** downloaded
- **6.4 million words**
- **32.8 MB** combined corpus
- Located in: `bible_corpus/bible_combined.txt`

## 🚀 Start Learning

**Important: Run from the project root directory** (where this file is located)

### For Windows Users:

The lessons use emojis and interactive prompts. Run them in PowerShell:

```powershell
# Set UTF-8 encoding (run once per session)
chcp 65001

# Run lesson 1
uv run python lessons/lesson_01_exploring_data.py
```

### For Mac/Linux Users:

```bash
uv run python lessons/lesson_01_exploring_data.py
```

## 📚 Lesson Path

1. `lessons/lesson_01_exploring_data.py` - Start here!
2. `lessons/lesson_02_tokenization.py`
3. `lessons/lesson_03_embeddings.py`
4. `lessons/lesson_04_attention.py`
5. `lessons/lesson_05_multihead_attention.py`

Each lesson is interactive - press Enter to move through sections.

## 🐛 Troubleshooting

**If emojis show as `?` boxes:**
- Windows: Run the PowerShell encoding command above
- Or use: `python run_lesson.py 1` (wrapper with UTF-8)

**If you see "Corpus not found":**
- Make sure you're in the project root directory
- The path should be: `bible_corpus/bible_combined.txt`

**If input() doesn't work:**
- You must run these in an interactive terminal
- They won't work if piped or run as background jobs

## 📊 What You Have

```
bible_data/          - 9 raw Bible files (34 MB)
bible_corpus/        - 10 cleaned files (67 MB)
  ├── bible_combined.txt  (all 9 combined - 34 MB)
  └── [individual cleaned files]
```

## 🎯 Ready to Start!

Open your terminal in this directory and run:

```bash
uv run python lessons/lesson_01_exploring_data.py
```

Happy learning! 🎓

# Publishing Checklist

This document outlines what to include/exclude when publishing this repository.

## ✅ Include (Safe to Publish)

### Root Files
- `README.md` - Main documentation
- `CLAUDE.md` - Claude instructions
- `LICENSE` - MIT License
- `.gitignore` - Git ignore rules

### Skills to Publish
- `skills/bewirtungsbeleg/` - German business meal receipt skill
  - ✅ `SKILL.md` - Skill documentation
  - ✅ `config.example.yml` - Configuration template
  - ✅ `pyproject.toml` - Python dependencies
  - ✅ `uv.lock` - Dependency lock file
  - ✅ `.gitignore` - Local gitignore
  - ✅ `scripts/create_bewirtungsbeleg.py` - Main script
  - ✅ `assets/signature.example.png` - Example signature reference
  - ✅ `references/steuerliche_anforderungen.md` - Tax requirements

### Documentation
- `context/` - All documentation files
  - ✅ `skills-overview.md`
  - ✅ `skills-best-practices.md`
  - ✅ `skills-quickstart.md`
  - ✅ `skills-claude-code.md`
  - ✅ `image-processing.md`

## ❌ Exclude (DO NOT Publish)

### Personal/Sensitive Data
- ❌ `skills/bewirtungsbeleg/config.yml` - YOUR personal config (gitignored)
- ❌ `skills/bewirtungsbeleg/assets/signature.png` - YOUR signature (gitignored)

### Other Skills (Not Ready for Publication)
- ❌ `skills/cropping-and-deskewing-receipts/`
- ❌ `skills/uploading-to-google-drive/`

### Test Files & Artifacts
- ❌ `*.pdf` - Generated PDFs (gitignored)
- ❌ `*.jpeg`, `*.jpg`, `*.png` - Test images in root (gitignored)
- ❌ `*.zip` - Archive files (gitignored)
- ❌ `.DS_Store` - macOS files (gitignored)
- ❌ `.venv/` - Virtual environments (gitignored)
- ❌ `__pycache__/` - Python cache (gitignored)

### Hidden Directories
- ❌ `.claude/` - Local Claude settings (contains symlinks with absolute paths) (gitignored)

## Building Skill Package (For Upload/Deployment)

To create a complete package of the bewirtungsbeleg skill for uploading to Claude:

```bash
# Using mise (recommended)
mise run build
```

This creates `dist/bewirtungsbeleg.zip` which **includes**:
- ✅ Your `config.yml` with personal details
- ✅ Your `signature.png`
- ✅ All skill code and documentation

The zip **excludes** only build artifacts (`.venv/`, `__pycache__/`, `.claude/`).

**Note:** This zip is for your personal use (uploading/deployment), not for public distribution. The git repository itself remains the source of truth for public sharing (with sensitive files gitignored).

## Pre-Publication Steps

1. **Verify gitignore is working:**
   ```bash
   git status
   ```
   Should NOT show:
   - `config.yml`
   - `assets/signature.png`
   - Any test PDFs or images
   - `.venv` directories

2. **Clean working directory:**
   ```bash
   # Remove test files
   rm -f *.pdf *.jpeg *.jpg *.png *.zip

   # Keep only signature.example.png in assets
   find skills/bewirtungsbeleg/assets -name "*.png" ! -name "signature.example.png" -delete
   ```

3. **Verify no sensitive data:**
   ```bash
   # Check for personal names
   grep -r "Sebastian\|Korfmann" skills/bewirtungsbeleg/ --exclude-dir=.venv --exclude=config.yml

   # Should only find it in config.yml (which is gitignored)
   ```

4. **Create clean branch for publishing:**
   ```bash
   git checkout -b publish
   git add .
   git commit -m "Prepare for publication"
   ```

5. **Final verification:**
   ```bash
   # Check what will be committed
   git ls-files

   # Verify config.yml and signature.png are NOT listed
   ```

## What Users Will Need to Do

After cloning the repository, users must:

1. Copy `config.example.yml` to `config.yml`
2. Edit `config.yml` with their details
3. Add their own `assets/signature.png`
4. Run `uv sync` to install dependencies

This ensures each user configures the skill with their own data.

## Repository Structure (Public View)

```
claude-skills/
├── README.md
├── CLAUDE.md
├── LICENSE
├── .gitignore
├── skills/
│   └── bewirtungsbeleg/
│       ├── SKILL.md
│       ├── config.example.yml        ← Template only
│       ├── pyproject.toml
│       ├── uv.lock
│       ├── .gitignore
│       ├── scripts/
│       │   └── create_bewirtungsbeleg.py
│       ├── assets/
│       │   └── signature.example.png  ← Example only
│       └── references/
│           └── steuerliche_anforderungen.md
└── context/
    ├── skills-overview.md
    ├── skills-best-practices.md
    ├── skills-quickstart.md
    ├── skills-claude-code.md
    └── image-processing.md
```

## Notes

- The `.gitignore` file ensures personal data (`config.yml`, `signature.png`) is never committed
- Template files (`config.example.yml`, `signature.example.png`) provide guidance
- All sensitive data remains local to each user's installation

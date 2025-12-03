# Claude Skills Repository

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/skorfmann/company-skills?style=social)](https://github.com/skorfmann/company-skills/stargazers)
[![GitHub last commit](https://img.shields.io/github/last-commit/skorfmann/company-skills)](https://github.com/skorfmann/company-skills/commits/main)
[![GitHub issues](https://img.shields.io/github/issues/skorfmann/company-skills)](https://github.com/skorfmann/company-skills/issues)

A repository showcasing custom Claude Skills with documentation and reference materials for building specialized capabilities.

## Table of Contents

- [Quick Start - Get Started in 5 Minutes](#quick-start---get-started-in-5-minutes)
- [What's Included](#whats-included)
  - [Skills](#skills)
  - [Context & Documentation](#context--documentation)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installing a Skill](#installing-a-skill)
  - [Using Skills with Claude](#using-skills-with-claude)
- [Development Tasks](#development-tasks)
- [What are Claude Skills?](#what-are-claude-skills)
- [Repository Structure](#repository-structure)
- [Creating Your Own Skills](#creating-your-own-skills)
- [Contributing](#contributing)
- [License](#license)
- [Resources](#resources)
- [Support](#support)

## Quick Start - Get Started in 5 Minutes

Want to try out the Bewirtungsbeleg skill right away? Follow these five simple steps:

### Step 1: Clone the Repository
```bash
git clone https://github.com/skorfmann/company-skills.git
cd company-skills/skills/bewirtungsbeleg
```

### Step 2: Set Up Configuration
```bash
cp config.example.yml config.yml
```
Edit `config.yml` and add your name/company:
```yaml
gastgeber: "Your Name / Your Company Name"
```

### Step 3: Add Your Signature
Place your signature image at `assets/signature.png` (see `assets/signature.example.png` for format reference).

### Step 4: Install Dependencies
```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### Step 5: Start Using the Skill
You're ready! The skill will now:
- Analyze restaurant receipt photos
- Extract information automatically
- Generate tax-compliant German Bewirtungsbeleg PDFs
- Merge receipts with signed expense forms

💡 **Tip:** Use with Claude Code CLI for automatic skill discovery, or run standalone using the provided scripts. See [Using Skills with Claude](#using-skills-with-claude) for details.

## What's Included

### Skills

#### Bewirtungsbeleg (German Business Meal Receipt Generator)
Creates German tax-compliant entertainment expense receipts (Bewirtungsbelege) from restaurant receipts with automatic signature and original receipt attachment.

**Features:**
- Analyzes restaurant receipts (photos, scans, or PDFs)
- Extracts key information automatically
- Generates tax-compliant Bewirtungsbeleg PDFs
- Merges original receipt with signed expense form
- Supports all common image formats with EXIF orientation correction

**Use when:** Creating formal business meal expense documentation for German tax purposes.

📁 Located in: `skills/bewirtungsbeleg/`

### Context & Documentation

The `context/` directory contains comprehensive documentation about Claude Skills:

- **skills-overview.md** - Architecture and concepts behind Claude Skills
- **skills-best-practices.md** - Guidelines for authoring effective skills
- **skills-quickstart.md** - Quick start guide for creating your first skill
- **skills-claude-code.md** - Integration with Claude Code CLI
- **image-processing.md** - Image processing techniques and best practices

## Getting Started

### Prerequisites

- Python 3.8+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- [mise](https://mise.jdx.dev/) (optional, for development tasks)

### Installing a Skill

1. Clone this repository:
   ```bash
   git clone https://github.com/skorfmann/company-skills.git
   cd company-skills
   ```

2. Navigate to the skill directory:
   ```bash
   cd skills/bewirtungsbeleg
   ```

3. Configure the skill:
   ```bash
   cp config.example.yml config.yml
   ```

4. Edit `config.yml` with your details:
   ```yaml
   gastgeber: "Your Name / Your Company Name"
   ```

5. Add your signature image:
   - Place your signature as `assets/signature.png`
   - See `assets/signature.example.png` for reference format

6. Install dependencies:
   ```bash
   uv sync
   # or with pip:
   # pip install -e .
   ```

### Using Skills with Claude

Skills are designed to work with Claude Code and other Claude interfaces:

1. **With Claude Code CLI:**
   - Skills are automatically discovered from the `skills/` directory
   - Claude loads skill instructions on-demand when needed
   - See `context/skills-claude-code.md` for details

2. **Standalone Usage:**
   - Each skill can also be used independently via its scripts
   - See individual skill documentation in their SKILL.md files

## Development Tasks

If you have [mise](https://mise.jdx.dev/) installed, you can use these convenience tasks:

### Setup the Skill

```bash
mise run setup
```

Automatically sets up the bewirtungsbeleg skill:
- Copies `config.example.yml` to `config.yml`
- Installs dependencies
- Prompts you to add your signature

### Build Distribution Package

```bash
mise run build
```

Creates a complete zip file of the bewirtungsbeleg skill in `dist/bewirtungsbeleg.zip`:
- **Includes** your `config.yml` and `signature.png` (for uploading/deployment)
- Excludes build artifacts and virtual environments (`.venv/`, `__pycache__/`, `.claude/`)
- Ready for uploading to Claude or deployment

### Clean Build Artifacts

```bash
mise run clean
```

Removes build directories and temporary files.

### Install Dependencies

```bash
mise run install
```

Installs Python dependencies using uv (or pip if uv is not available).

## What are Claude Skills?

Skills are modular capabilities that extend Claude's functionality through:

- **SKILL.md** - Instructions and workflows (loaded on-demand)
- **Scripts** - Executable code for deterministic operations
- **References** - Supporting documentation and resources

### Key Concepts

**Progressive Loading**: Skills use a three-level loading system to minimize context usage:
1. **Metadata** (always loaded) - Name and description from YAML frontmatter
2. **Instructions** (when triggered) - SKILL.md content loaded via filesystem
3. **Resources** (as needed) - Additional files accessed only when referenced

**Filesystem-Based**: Skills exist as directories accessed through bash commands, enabling Claude to load only what's needed for each task.

## Repository Structure

```
claude-skills/
├── README.md                          # This file
├── CLAUDE.md                          # Repository instructions for Claude
├── skills/                            # Custom skills
│   └── bewirtungsbeleg/              # Business meal receipt skill
│       ├── SKILL.md                  # Skill instructions
│       ├── config.example.yml        # Configuration template
│       ├── config.yml                # Your config (gitignored)
│       ├── pyproject.toml            # Python dependencies
│       ├── scripts/                  # Executable scripts
│       ├── assets/                   # Resources (signatures, etc.)
│       └── references/               # Documentation
└── context/                          # Skills documentation
    ├── skills-overview.md
    ├── skills-best-practices.md
    ├── skills-quickstart.md
    ├── skills-claude-code.md
    └── image-processing.md
```

## Creating Your Own Skills

Want to create your own skill? Follow these steps:

1. Read `context/skills-quickstart.md` for a quick introduction
2. Review `context/skills-best-practices.md` for authoring guidelines
3. Use the Bewirtungsbeleg skill as a reference implementation
4. Structure your skill with:
   - SKILL.md with YAML frontmatter (name + description)
   - Scripts for deterministic operations
   - References for supporting documentation

## Contributing

Contributions are welcome! Please:

1. Follow the structure and conventions in existing skills
2. Include comprehensive SKILL.md documentation
3. Add configuration templates for any personal data
4. Test your skill thoroughly before submitting

## License

MIT License - see LICENSE file for details

## Resources

- [Claude Code Documentation](https://docs.claude.com/claude-code)
- [Skills Best Practices](context/skills-best-practices.md)
- [Skills Overview](context/skills-overview.md)

## Support

For issues or questions:
- Open an issue on GitHub
- Check the documentation in the `context/` directory
- Review existing skills for examples

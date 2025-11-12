# Claude Skills Repository

A repository showcasing custom Claude Skills with documentation and reference materials for building specialized capabilities.

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
   git clone https://github.com/yourusername/claude-skills.git
   cd claude-skills
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

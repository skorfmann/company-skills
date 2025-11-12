# Claude Skills Repository

This repository contains custom Claude Skills and reference materials for building specialized capabilities.

## Structure

- `skills/` - Custom skill implementations
- `context/` - Claude Skills documentation and best practices

## What are Skills?

Skills are modular capabilities that extend Claude's functionality through:
- **SKILL.md** - Instructions and workflows (loaded on-demand)
- **Scripts** - Executable code for deterministic operations
- **References** - Supporting documentation and resources

## Key Concepts

**Progressive Loading**: Skills use a three-level loading system to minimize context usage:
1. **Metadata** (always loaded) - Name and description from YAML frontmatter
2. **Instructions** (when triggered) - SKILL.md content loaded via filesystem
3. **Resources** (as needed) - Additional files accessed only when referenced

**Filesystem-Based**: Skills exist as directories accessed through bash commands, enabling Claude to load only what's needed for each task.

## Working with Skills

When creating or modifying skills in this repo:
- Follow the structure in `context/skills-best-practices.md`
- Each skill requires SKILL.md with YAML frontmatter (name + description)
- Keep instructions clear and task-oriented
- Use scripts for repetitive or complex operations
- Store references separately to reduce context usage

## References

See `context/` directory for complete documentation on:
- Skills overview and architecture
- Best practices for authoring
- Quickstart guide

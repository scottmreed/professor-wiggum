# Manuscript Updater Skill

Automated workflow for maintaining the mechanistic manuscript (`manuscript/mechanistic_manuscript.rtf`) by integrating repository changes, competition updates, literature surveillance, and figure management.

## Overview

This skill provides a comprehensive solution for keeping scientific manuscripts current with project evolution. It monitors:

- **Repository Changes**: Training runs, PR contributions, performance improvements, and ralph loop progress
- **Competition Updates**: Clawdiator leaderboard changes and benchmark results
- **Literature**: New relevant papers in mechanistic chemistry and AI
- **Figures**: Automated figure generation and version tracking
- **Attribution**: Proper credit for contributors and authors

## Components

### Core Tools

1. **`git_monitor.py`** - Monitors repository changes and extracts manuscript-relevant information
2. **`clawdiator_scraper.py`** - Scrapes competition leaderboard and benchmark results
3. **`literature_monitor.py`** - Searches for and evaluates new academic papers
4. **`manuscript_integrator.py`** - Handles RTF manipulation and content integration
5. **`figure_handler.py`** - Manages manuscript figures and PNG generation
6. **`workflow.py`** - Orchestrates the complete update process
7. **`add_harness_figure.py`** - Script to add the harness architecture figure

### Key Features

- **Automated Attribution**: Properly credits PR authors and contributors
- **Multi-Source Integration**: Combines repo changes, competition results, literature, and figures
- **RTF Preservation**: Maintains document formatting while adding content
- **Figure Management**: Automated PNG generation from Mermaid diagrams with version tracking
- **Quality Control**: Validates manuscript integrity and scientific standards
- **Flexible Content Management**: Supports addition, modification, and removal of content
- **Update Checking**: Determines when manuscript updates are needed without applying them

## Usage

### Quick Start

```bash
# Navigate to the skill directory
cd skills/manuscript_updater

# Check what updates are needed
python workflow.py --check-updates

# Run a dry-run update to see what changes would be made
python workflow.py

# Apply the updates to the manuscript
python workflow.py --apply

# Add the harness architecture figure
python add_harness_figure.py

# Check manuscript integrity
python workflow.py --validate

# View update history
python workflow.py --history
```

### Individual Tool Testing

```bash
# Test repository monitoring
python git_monitor.py

# Test competition scraping
python clawdiator_scraper.py

# Test literature search
python literature_monitor.py

# Test figure handling
python figure_handler.py --generate-harness --status

# Test manuscript integration
python manuscript_integrator.py
```

## Configuration

### Manuscript Path
Default: `manuscript/mechanistic_manuscript.rtf`

Override with:
```bash
python workflow.py --manuscript path/to/your/manuscript.rtf
```

### Figures Directory
Default: `manuscript/figures`

Figures are stored as PNG files with corresponding Mermaid source files for version control.

### Search Parameters

- **Repository**: Monitors last 30 days by default
- **Literature**: Searches last 6 months for relevant papers
- **Competition**: Scrapes current leaderboard standings
- **Figures**: Checks for updates based on source file modifications

## Manuscript Sections

The skill maintains these standard academic sections:

- **Abstract**: High-level system description and evolution
- **Introduction**: Background, motivation, and system overview
- **Methods**: Technical implementation, architecture, and figures
- **Results**: Performance metrics, evaluation results, comparisons
- **Discussion**: Interpretation, limitations, future directions
- **Related Work**: Literature review and competitive analysis
- **Conclusions**: Summary and broader implications
- **Acknowledgments**: Contributor attribution and support
- **References**: Complete citation list

## Figure Management

### Supported Figure Types

1. **Harness Architecture Diagram** (Figure 1)
   - Automatically generated from Mermaid diagram
   - Shows the modular pipeline and validation flow
   - Updated when harness structure changes

### Figure Generation Workflow

```bash
# Generate harness figure
python figure_handler.py --generate-harness

# Check figure status
python figure_handler.py --status

# Check for needed figure updates
python figure_handler.py --check-updates
```

### Figure Registry

Located at: `manuscript/figures/figure_registry.json`

Tracks:
- Figure versions and last update times
- File hashes for change detection
- Descriptions and metadata

## Update Types

### Repository Updates
- **Commits**: Significant code changes affecting methodology
- **PRs**: Author attribution for substantive contributions
- **Training Results**: Performance metric updates
- **Ralph Loops**: Convergence and optimization progress

### Competition Updates
- **Leaderboard Changes**: Ranking and score updates
- **Methodology Shifts**: New competitive approaches
- **Benchmark Evolution**: Changes in evaluation standards

### Literature Updates
- **High-Relevance Papers**: Direct methodological contributions
- **Survey Articles**: Comprehensive field overviews
- **Benchmark Papers**: New evaluation standards

### Figure Updates
- **Source Changes**: Regenerate PNG when Mermaid files change
- **Missing Figures**: Identify unreferenced figures
- **Version Tracking**: Maintain figure version history

## Update Checking

The `--check-updates` flag provides a non-destructive way to see what updates are needed:

```bash
python workflow.py --check-updates
```

This checks:
- Figure update requirements
- Recent repository changes
- Competition leaderboard changes
- New literature publications

## Attribution System

### Automatic Attribution
- **PR Authors**: Identified from GitHub PR metadata
- **Co-Authors**: Extracted from commit co-author trailers
- **Contributors**: Added to acknowledgments section

### Attribution Types
- **Technical**: Algorithm or implementation contributions
- **Methodological**: Research design or approach contributions
- **General**: Documentation, testing, or infrastructure contributions

## Quality Assurance

### Content Management
- **Addition Criteria**: Measurable improvements or significant changes
- **Removal Criteria**: Outdated results replaced by current data
- **Preservation**: Maintains scientific rigor and academic standards

### Validation Checks
- **RTF Integrity**: Document structure and formatting preserved
- **Section Structure**: Standard academic sections maintained
- **Figure References**: All referenced figures exist and are current
- **Citation Format**: Consistent reference formatting
- **Content Coherence**: Logical flow and scientific accuracy

## Integration with Development Workflow

### Automated Triggers
- **Post-Merge**: Repository changes trigger update checks
- **Weekly Review**: Scheduled comprehensive literature and competition review
- **Release Preparation**: Manuscript updates before version releases
- **Figure Updates**: Regenerate figures when harness changes

### Manual Triggers
- **Ad-hoc Updates**: Run workflow when significant changes occur
- **Literature Reviews**: Periodic comprehensive literature surveys
- **Competition Monitoring**: Check for leaderboard changes
- **Figure Maintenance**: Update diagrams when architecture evolves

## Output and Logging

### Update Log
Located at: `manuscript/update_log.json`

Contains:
- Timestamp of each update run
- Changes identified and applied
- Attribution information
- Validation results
- Figure update status

### Figure Registry
Located at: `manuscript/figures/figure_registry.json`

Contains:
- Figure metadata and versions
- File hashes and timestamps
- Update history

### Backup System
- **Automatic Backups**: Created before each manuscript modification
- **Backup Location**: `manuscript/backups/`
- **Naming**: `manuscript_backup_YYYYMMDD_HHMMSS.rtf`

## Dependencies

- **Python 3.8+**
- **requests** (for web scraping)
- **beautifulsoup4** (for HTML parsing)
- **Git** (for repository monitoring)
- **Mermaid CLI** (optional, for PNG generation)

## Troubleshooting

### Common Issues

**RTF Parsing Errors**
- Ensure manuscript file is valid RTF format
- Check for corrupted RTF control codes
- Validate with RTF editor before running updates

**Figure Generation Failures**
- Install Mermaid CLI: `npm install -g @mermaid-js/mermaid-cli`
- Check Node.js and Puppeteer installation
- Verify Mermaid diagram syntax

**Missing Updates**
- Verify repository is properly initialized
- Check network connectivity for external data sources
- Review date ranges for change detection

**Attribution Errors**
- Ensure PR descriptions include contributor information
- Check git configuration for proper author metadata
- Verify co-author trailer format in commits

### Recovery
- **Backup Restoration**: Copy from `manuscript/backups/` directory
- **Log Review**: Check `update_log.json` for change history
- **Manual Revert**: Use git to revert manuscript changes
- **Figure Recreation**: Regenerate from Mermaid sources

## Contributing

### Extending the Skill
- **New Data Sources**: Add scrapers for additional competitions or databases
- **Enhanced Attribution**: Improve contributor identification algorithms
- **Additional Figures**: Create new Mermaid diagrams for other manuscript figures
- **Content Analysis**: Add more sophisticated manuscript content analysis

### Testing
- **Dry Runs**: Always test with `--dry-run` first
- **Validation**: Use `--validate` to check manuscript integrity
- **Figure Checks**: Use `--check-updates` to verify figure status
- **Backup Review**: Verify backups before applying changes

## License and Attribution

This skill is part of the Mechanistic project. Please attribute contributors according to the project's contribution guidelines.
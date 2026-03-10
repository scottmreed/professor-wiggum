---
skill_type: project
call_name: manuscript_updater
kind: workflow
phase: maintenance
version: 1
---

# Manuscript Updater

Automated workflow for maintaining the mechanistic manuscript (`manuscript/mechanistic_manuscript.rtf`) by integrating repository changes, competition updates, and relevant literature. This skill ensures the manuscript stays current with project evolution while maintaining scientific rigor and proper attribution.

## Scope

This skill monitors repository activity and external sources to automatically suggest manuscript updates. It handles:

- Training run completions and performance changes
- PR contributions with author attribution
- Ralph loop iterations and improvements
- Clawdiator competition leaderboard updates
- New relevant literature identification and integration

## Tools and Capabilities

### Repository Monitoring Tools

**Git Change Analysis**
- Monitors commits, PRs, and merges since last manuscript update
- Extracts training run results, evaluation scores, and performance metrics
- Identifies author contributions requiring attribution
- Tracks ralph loop iterations and convergence patterns

**Performance Data Extraction**
- Parses leaderboard updates and evaluation results
- Extracts key metrics (accuracy, step count, validation rates)
- Identifies significant improvements or regressions
- Flags results that warrant manuscript discussion

### External Data Integration Tools

**Clawdiator Scraper**
- Monitors public evaluation competition leaderboard
- Extracts benchmark scores and methodology comparisons
- Identifies competitive positioning changes
- Flags new baseline or state-of-the-art results

**Literature Surveillance**
- Searches recent publications in mechanistic chemistry and AI
- Identifies papers setting precedents for similar research
- Extracts key findings, methodologies, and citations
- Prioritizes papers with direct relevance to the Mechanistic approach

### Manuscript Integration Tools

**Content Management**
- Maintains section structure with standard academic headings
- Supports text addition, modification, and removal over time
- Ensures consistent citation formatting and reference management
- Preserves RTF formatting and document structure

**Attribution System**
- Tracks all contributors mentioned in PRs and commits
- Maintains author contribution statements
- Updates acknowledgments and author lists as needed
- Ensures proper credit for intellectual contributions

## Workflow

### 1. Repository Analysis Phase
- Scan recent commits and PRs for substantive changes
- Extract performance data from training runs and evaluations
- Identify contributors requiring manuscript attribution
- Flag ralph loop progress and convergence metrics

### 2. External Data Collection Phase
- Scrape Clawdiator leaderboard for competition updates
- Search academic databases for relevant new literature
- Cross-reference findings with current manuscript content
- Prioritize updates based on impact and relevance

### 3. Content Integration Phase
- Propose specific manuscript sections for updates
- Generate attribution statements for new contributors
- Suggest literature citations and discussion points
- Identify outdated content for removal or revision

### 4. Quality Assurance Phase
- Verify RTF formatting preservation
- Check citation consistency and accuracy
- Validate scientific claims against evidence
- Ensure manuscript coherence and flow

## Standard Section Headings

The skill maintains these standard manuscript sections:

- **Abstract**: High-level summary of system capabilities and evolution
- **Introduction**: Background, motivation, and system overview
- **Methods**: Technical implementation details and architecture
- **Results**: Performance metrics, evaluation results, and comparisons
- **Discussion**: Interpretation of results, limitations, and future directions
- **Related Work**: Literature review and competitive analysis
- **Conclusions**: Summary and broader implications
- **Acknowledgments**: Contributor attribution and funding
- **References**: Complete citation list

## Content Management Guidelines

### Addition Criteria
- **Training Results**: Include when performance changes exceed 5% threshold
- **PR Contributions**: Add when introducing novel capabilities or significant improvements
- **Competition Updates**: Include when changing competitive positioning
- **Literature**: Add papers with direct methodological or conceptual relevance

### Removal Criteria
- **Outdated Results**: Remove superseded performance metrics after 3 months
- **Preliminary Data**: Replace provisional results with final validated data
- **Redundant Content**: Consolidate overlapping discussions
- **Scope Changes**: Remove content no longer relevant to current system focus

### Attribution Standards
- **Primary Authors**: Full names with institutional affiliations
- **Contributors**: Name recognition in acknowledgments section
- **PR Authors**: Attribution for substantive technical contributions
- **Funding Sources**: Clear acknowledgment of support sources

## Output Format

The skill generates structured update proposals in the following format:

```json
{
  "timestamp": "2026-03-10T12:00:00Z",
  "sections_to_update": [
    {
      "section": "Results",
      "action": "add|modify|remove",
      "content": "Specific manuscript text",
      "justification": "Evidence-based rationale",
      "citations": ["references to add"],
      "attributions": ["contributors to acknowledge"]
    }
  ],
  "new_references": [
    {
      "key": "Author2026",
      "full_citation": "Complete APA/MLA citation",
      "relevance": "Why this paper is relevant"
    }
  ],
  "rtf_formatting_notes": [
    "Specific RTF formatting requirements"
  ]
}
```

## Integration with Development Workflow

This skill integrates with the project's evolutionary development process:

1. **Post-Merge Hook**: Automatically triggered after PR merges
2. **Weekly Review**: Scheduled analysis of accumulated changes
3. **Release Preparation**: Comprehensive manuscript updates before releases
4. **Competition Monitoring**: Daily checks for leaderboard changes

## Quality Controls

- **Evidence Gating**: All updates must be supported by repository data or external sources
- **Review Process**: Generated updates undergo human review before manuscript changes
- **Version Control**: All manuscript changes tracked in git with clear commit messages
- **Backup Preservation**: Previous manuscript versions archived for reference

## Notes

- This skill is designed for continuous manuscript maintenance rather than one-time updates
- All changes preserve the scientific integrity and academic standards of the manuscript
- The skill prioritizes accuracy over completeness when faced with conflicting information
- RTF formatting constraints are respected to maintain document compatibility

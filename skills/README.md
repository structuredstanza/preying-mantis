# Skills

Each directory is one skill: a `SKILL.md` an agent loads on demand, with any
`references/` it needs.

| Skill                                                         | What it does                                                                                                                                               |
| ------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`mantis-architecture`](mantis-architecture/SKILL.md)         | Synthesizes raw learnings and codebase analysis into an interlinked Markdown Knowledge Base (KB).                                                          |
| [`mantis-calibrate`](mantis-calibrate/SKILL.md)               | Calculates the final risk score based on empirical evidence and architectural impact.                                                                      |
| [`mantis-chain`](mantis-chain/SKILL.md)                       | Analyzes individual security findings to identify and construct complex exploit chains.                                                                    |
| [`mantis-critic`](mantis-critic/SKILL.md)                     | Assesses the production viability of findings, filtering out debug-only features and assertion traps.                                                      |
| [`mantis-dedupe`](mantis-dedupe/SKILL.md)                     | Consolidates raw security findings to eliminate redundant reports.                                                                                         |
| [`mantis-history`](mantis-history/SKILL.md)                   | Analyzes the repository's version control system (VCS) history to extract past vulnerabilities, security fixes, and vulnerability patterns.                |
| [`mantis-meta-agent`](mantis-meta-agent/SKILL.md)             | Acts as the persistent supervisor, launching and monitoring the automated review campaign.                                                                 |
| [`mantis-pipeline-adapter`](mantis-pipeline-adapter/SKILL.md) | Interactively guides the design and implementation of custom deterministic orchestrator harnesses.                                                         |
| [`mantis-plan`](mantis-plan/SKILL.md)                         | Formulates a targeted defensive security reviewing plan based on the active threat model and historical learnings.                                         |
| [`mantis-reflect`](mantis-reflect/SKILL.md)                   | Extracts learnings from execution trajectories at the end of a Mantis loop.                                                                                |
| [`mantis-report`](mantis-report/SKILL.md)                     | Generates a human-readable security review packet compiled from confirmed findings and exploit chains.                                                     |
| [`mantis-reproduce`](mantis-reproduce/SKILL.md)               | Generates and runs crash reproducers to verify security flaws.                                                                                             |
| [`mantis-researcher`](mantis-researcher/SKILL.md)             | Audits production source code files based on the strategy in workspace/plan.json.                                                                          |
| [`mantis-review`](mantis-review/SKILL.md)                     | Independently reviews findings and filters out false positives.                                                                                            |
| [`mantis-structural-index`](mantis-structural-index/SKILL.md) | Builds a content-addressed semantic-unit index from source code for structural context.                                                                    |
| [`mantis-summarize`](mantis-summarize/SKILL.md)               | Pre-processes the repository by generating security-focused summaries (mantis-summary.md) for each directory to make planning and research more efficient. |
| [`mantis-threat-model`](mantis-threat-model/SKILL.md)         | Synthesizes trust boundaries, attack surfaces, and attacker profiles into a living threat model.                                                           |

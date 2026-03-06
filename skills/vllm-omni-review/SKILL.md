---
name: vllm-omni-review
description: Use when reviewing PRs on vllm-project/vllm-omni. Triggers domain-specific skills based on PR type for context-aware reviews focused on tests, evidence, and critical issues.
---

# vLLM-Omni PR Review

## Overview

Review pull requests for [vLLM-Omni](https://github.com/vllm-project/vllm-omni) by leveraging domain-specific skills. Focus on critical issues: missing tests, unvalidated claims, security, design flaws, and breaking changes.

## Review Constraints

| Constraint | Value |
|------------|-------|
| Max inline comments | 5 per PR |
| Comment length | 2-4 sentences each |
| Small doc fix | 0 comments expected |
| Large feature | 3-5 comments on critical gaps |

### Banned Phrases (Generic Praise)

- "solid", "generally", "looks good", "well done", "nice work", "great job"
- "comprehensive", "well structured", "good implementation"
- Any phrase without specific code location reference

## Review Workflow

### Step 0: Verify CI Gate (REQUIRED FIRST)

**Check these BEFORE reviewing code. If any fail, post a gate-failure comment and STOP.**

```bash
gh pr view <pr_number> --repo vllm-project/vllm-omni --json mergeable,statusCheckRollup --jq '{mergeable, checks: [.statusCheckRollup[] | {name, conclusion}]}'
```

| Check | Passing State | Action if Failed |
|-------|---------------|------------------|
| DCO | `SUCCESS` | Comment "DCO check failed. Please sign your commits with `git commit -s`." |
| pre-commit | `SUCCESS` | Comment "Pre-commit checks failed. Run `pre-commit run --all-files` and fix issues." |
| mergeable | `MERGEABLE` | Comment "Merge conflicts detected. Please rebase onto main and resolve conflicts." |

**Gate failure comment template:**
```
🚫 **Review Blocked: CI Gate Failed**

- [ ] DCO: <status>
- [ ] Pre-commit: <status>
- [ ] Merge conflicts: <status>

Please resolve these issues before review proceeds.
```

**Only proceed to Step 1 when all gates pass.**

### Step 1: Fetch PR Data

```bash
gh pr view <pr_number> --repo vllm-project/vllm-omni --json title,body,author,state,files,closingIssuesReferences
gh pr diff <pr_number> --repo vllm-project/vllm-omni
```

### Step 1.5: Fetch Linked Context (Issues & Related PRs)

**Fetch linked issues for context:**

```bash
# Get closing issues referenced in PR body (Fixes #123, Closes #456, etc.)
gh pr view <pr_number> --repo vllm-project/vllm-omni --json closingIssuesReferences --jq '.closingIssuesReferences[] | {number, title, body}'

# For each linked issue, get full details
gh issue view <issue_number> --repo vllm-project/vllm-omni --json title,body,labels,state,comments
```

**Fetch related PRs touching the same files:**

```bash
# Get list of changed files from PR
gh pr view <pr_number> --repo vllm-project/vllm-omni --json files --jq '.files[].path'

# For each significant file, find other open/recent PRs
gh search prs --repo vllm-project/vllm-omni "<file_path>" --merged --limit 5 --json number,title,author,closedAt
```

**What linked context provides:**

| Context Type | Value for Review |
|--------------|------------------|
| **Linked Issues** | Original bug description, acceptance criteria, design decisions |
| **Related Merged PRs** | Patterns to follow, prior solutions, area-specific conventions |
| **Issue Labels** | Priority (P0/P1), area owners, breaking change flags |
| **Issue Comments** | Prior discussions, rejected approaches, constraints |

**When linked context is critical:**

- `[Bugfix]` PRs → **Always fetch linked issue** for bug reproduction steps
- `[Feature]` PRs → **Always fetch linked issue** for acceptance criteria
- Complex multi-file changes → **Check related PRs** for area conventions
- Changes to shared infrastructure → **Check related PRs** for coordination needs

**Limits:** 2-3 linked issues max, 5 related PRs per file max

### Step 2: Identify PR Type & Trigger Skill

Check PR title prefix against the table below. If domain-specific context is needed, invoke the corresponding skill.

| PR Prefix | Trigger Skill | Key Focus |
|-----------|---------------|-----------|
| `[Image]`, `[ImageGen]` | `vllm-omni-image-gen` | Generation quality, memory |
| `[Video]`, `[VideoGen]` | `vllm-omni-video-gen` | Temporal consistency, VRAM |
| `[Audio]`, `[TTS]` | `vllm-omni-audio-tts` | Audio quality, latency |
| `[Multimodal]` | `vllm-omni-multimodal` | Cross-modal alignment |
| `[Distributed]` | `vllm-omni-distributed` | Scaling, disaggregation |
| `[Quantization]` | `vllm-omni-quantization` | Accuracy loss, compatibility |
| `[Performance]` | `vllm-omni-perf` | Benchmarks, claims validation |
| `[Hardware]` | `vllm-omni-hardware` | Backend compatibility |
| `[API]` | `vllm-omni-api` | OpenAI compat, input validation |
| `[CI]` | `vllm-omni-cicd` | Pipeline correctness |
| `[Model]` | `vllm-omni-contrib` | Integration patterns |
| `[Bugfix]` | — | Regression test required |
| `[Refactor]` | — | No behavior change |
| `[Feature]` | — | Tests + docs required |

**When to invoke skill:**
- Diff content is unclear without domain context
- Domain-specific validation needed (e.g., model architecture, API contract)
- Performance claims need verification
- **Change affects a specialized subsystem** (CPU offloading, tensor parallelism, diffusion sampling, quantization methods)
- **Even if PR provides benchmark data**, invoke skill when the change is in a specialized area

**When NOT to invoke:**
- Simple refactors with clear intent
- Documentation-only changes
- Already have sufficient context from diff
- PR touches only one subsystem with obvious changes

### Multi-Category PRs

When a PR has multiple prefixes (e.g., `[Perf][Distributed]`):

1. **Identify primary type** (first prefix usually indicates main intent)
2. **Invoke primary skill first**
3. **Use secondary skill for cross-cutting concerns:**

| Scenario | Primary | Secondary |
|----------|---------|----------|
| `[Perf]` + `[Distributed]` | `vllm-omni-perf` | `vllm-omni-distributed` (check TP scaling) |
| `[Perf]` + `[Model]` | `vllm-omni-perf` | `vllm-omni-contrib` (check model integration) |
| `[Model]` + `[Quantization]` | `vllm-omni-contrib` | `vllm-omni-quantization` (check quality impact) |
| `[Feature]` + `[API]` | `vllm-omni-api` | — (API skill usually sufficient) |
| `[Distributed]` + `[Hardware]` | `vllm-omni-distributed` | `vllm-omni-hardware` (check backend compat) |

### Hardware Platform Detection

**Scan the diff for hardware-specific patterns. If found, invoke `vllm-omni-hardware` skill.**

```bash
gh pr diff <pr_number> --repo vllm-project/vllm-omni | grep -E 'is_npu|is_cuda|torch\.cuda|torch_npu'
```

| Pattern | Platform | Review Focus |
|---------|----------|--------------|
| `is_npu` | NPU (Ascend) | NPU-specific ops, memory, compatibility |
| `is_cuda` | CUDA | Device checks, conditional paths |
| `torch.cuda` | CUDA | Device placement, memory management |
| `torch_npu` | NPU (Ascend) | NPU bindings, compatibility |

**When detected, check for:**
- Missing NPU fallback paths when CUDA is primary
- Hardcoded device assumptions (`cuda:0` vs device-agnostic)
- Platform-specific ops without guards
- Memory management differences between platforms

### Step 3: Run Red Flag Checks

**Required for ALL PRs:**

- [ ] New API without tests?
- [ ] New model without tests?
- [ ] Performance claims without benchmarks?
- [ ] Mixin after `nn.Module` with `__init__` setting attributes?
- [ ] API changes without documentation?

### Step 4: Check Pitfalls by Directory

Check pitfalls relevant to affected directories first:

 then consult [references/pitfalls.md](references/pitfalls.md) for detailed explanations.

| Affected Directory | Pitfalls to Check |
|--------------------|-------------------|
| `vllm_omni/engine/` | Scheduler state, pipeline coordination |
| `vllm_omni/diffusion/` | Memory growth, generation quality |
| `vllm_omni/connectors/` | Shared memory leaks, IPC issues |
| `vllm_omni/stages/` | Stage lifecycle, config validation |
| `vllm_omni/model_executor/` | Weight loading, device placement |
| `vllm_omni/entrypoints/` | Input validation, error handling |
| `*` (any) | MRO issues with mixins, async vs sync path differences |

- MRO issues with mixins
- Connector state management
- Async vs Sync path differences
- Stage configuration validation

### Step 5: Post Review

```bash
gh api repos/vllm-project/vllm-omni/pulls/<pr_number>/reviews --input - <<EOF
{
  "event": "REQUEST_CHANGES" | "APPROVE" | "COMMENT",
  "body": "<summary>",
  "comments": [
    {"path": "<file>", "line": <num>, "body": "<comment>"}
  ]
}
EOF
```

## Comment Phrasing Guidelines

| PR State | Style |
|----------|-------|
| **Draft** | Ask questions, suggest alternatives: "Consider X for Y because..." |
| **Ready** | Request changes: "Please address..." "This needs..." |
| **Approved/In Progress** | Only comment if blocking: "Note: found issue at..." |

## Priority Order

1. **Missing tests** — highest priority
2. **Unvalidated claims** — demand measurements/evidence
3. **Security concerns** — input validation, resource exhaustion
4. **Design flaws** — architectural issues, race conditions
5. **Breaking changes** — undocumented API changes

**Skip:** Minor style issues, nitpicks, nice-to-haves, linter-covered issues

## Critical Directories

| Directory | Impact | Review Focus |
|-----------|--------|--------------|
| `vllm_omni/engine/` | **Critical** | Scheduler, pipeline coordination |
| `vllm_omni/model_executor/` | **Critical** | Model loading, weight management |
| `vllm_omni/connectors/` | High | Shared memory, IPC |
| `vllm_omni/entrypoints/` | High | API validation, error handling |
| `vllm_omni/stages/` | High | Stage lifecycle, state management |
| `vllm_omni/diffusion/` | High | Generation quality, memory |

## Example Comments

**Good (Demands Evidence):**
```
Where are the memory measurements? The PR claims "50% reduction" but provides no before/after data. Run benchmarks with realistic workloads and report peak VRAM usage.
```

**Good (Missing Tests):**
```
Missing regression test for this bug fix. Add a test that reproduces the original issue and verifies this fix prevents it.
```

**Good (MRO Issue):**
```
This mixin is listed after nn.Module but has an __init__ that sets attributes. When nn.Module.__init__ is called, the mixin's __init__ won't run. Use lazy initialization with @property instead.
```

**Bad (Generic):**
```
The implementation looks good. Consider adding tests.
```

## Context Fetching

**When to fetch more context:**
- New imports with performance/compatibility implications
- 3-line diff context isn't enough
- Code changes might require config updates
- Need to understand prior decisions or constraints

**Tools:**
```bash
# Get surrounding code
gh api repos/vllm-project/vllm-omni/contents/<path>?ref=<branch>

# Find symbol definition
gh search code --repo vllm-project/vllm-omni "class <SymbolName>"

# Check related configs
gh search code --repo vllm-project/vllm-omni "<config_key>" --extension yaml

# Find discussions on similar topics
gh search discussions --repo vllm-project/vllm-omni "<topic>"
```

**Limits:** 3-5 context fetches per review, 20-50 lines each

> **Note:** For linked issues and related PRs, see Step 1.5 which handles this systematically.

## Known Dependencies

Do NOT flag these as missing:
- `einops` — inherited from vLLM
- `diffusers` — already in requirements

## References

- [Common Pitfalls](references/pitfalls.md) — MRO issues, connector state, async patterns
- [Architecture](references/architecture.md) — System overview and critical paths
- [Code Patterns](references/code-patterns.md) — Async, distributed, KV cache patterns

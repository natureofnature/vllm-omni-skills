#!/usr/bin/env python3
"""
Hybrid skill updater - Plan C implementation.

Three-layer update mechanism:
1. SKILL.md - Only update important and persistent content (tables, examples, troubleshooting)
2. docs/updates/{skill}.md - Record all updates (detailed log)
3. docs/CHANGELOG.md - Record important updates (centralized summary)

Archive cycle: 4 weeks (aligned with vllm-omni release)
"""

import json
import re
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


class HybridSkillUpdater:
    """混合方案C：三层更新机制"""

    def __init__(self, workspace_root: str):
        self.workspace = Path(workspace_root)
        self.vllm_omni_repo = "vllm-project/vllm-omni"
        self.skills_dir = self.workspace / "vllm-omni-skills" / "skills"
        self.updates_dir = self.workspace / "vllm-omni-skills" / "docs" / "updates"
        self.changelog_file = self.workspace / "vllm-omni-skills" / "docs" / "CHANGELOG.md"

    def get_recent_prs(self, since_days: int = 7) -> list[dict]:
        """获取最近的merged PRs"""
        cmd = [
            "gh", "pr", "list",
            "--repo", self.vllm_omni_repo,
            "--state", "merged",
            "--limit", "50",
            "--json", "number,title,body,labels,mergedAt,files"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Error fetching PRs: {result.stderr}")
            return []

        prs = json.loads(result.stdout)

        # 按日期过滤
        cutoff = datetime.now() - timedelta(days=since_days)
        recent_prs = []
        for pr in prs:
            merged_at = datetime.fromisoformat(pr["mergedAt"].replace("Z", "+00:00"))
            if merged_at.replace(tzinfo=None) > cutoff:
                recent_prs.append(pr)

        return recent_prs

    def categorize_change(self, pr: dict) -> list[str]:
        """判断PR影响哪些skills"""
        categories = []
        title = pr.get("title", "").lower()
        body = pr.get("body", "").lower()
        files = [f["path"] for f in pr.get("files", [])]

        # 新模型支持
        if any(kw in title for kw in ["new model", "add model", "support for"]):
            if any(kw in title for kw in ["image", "flux", "sd", "hunyuan"]):
                categories.append("vllm-omni-image-gen")
            if any(kw in title for kw in ["video", "wan", "cogvideo"]):
                categories.append("vllm-omni-video-gen")
            if any(kw in title for kw in ["audio", "tts"]):
                categories.append("vllm-omni-audio-tts")

        # API变更
        if any(f.startswith("vllm_omni/entrypoints/") for f in files):
            categories.append("vllm-omni-api")

        # 量化
        if any(kw in title or kw in body for kw in ["quantiz", "awq", "gptq", "fp8"]):
            categories.append("vllm-omni-quantization")

        # 性能优化
        if any(kw in title for kw in ["performance", "speed", "optimization", "cache"]):
            categories.append("vllm-omni-perf")

        # CI/CD
        if any(f.startswith(".github/") or f.startswith("tests/") for f in files):
            categories.append("vllm-omni-cicd")

        return list(set(categories))

    def is_important_update(self, pr: dict) -> bool:
        """判断是否是重要更新（需要写入SKILL.md）"""
        title = pr.get("title", "").lower()

        # 新模型
        if any(kw in title for kw in ["new model", "add model", "support for"]):
            return True

        # 重要特性
        if any(kw in title for kw in ["[feature]", "[feat]"]):
            return True

        # 关键fix
        if any(kw in title for kw in ["crash", "error", "fix load", "fix init"]):
            return True

        return False

    def update_skill_core(self, skill_name: str, pr: dict):
        """更新SKILL.md的核心内容（仅重要更新）"""

        skill_file = self.skills_dir / skill_name / "SKILL.md"
        if not skill_file.exists():
            print(f"  ⚠️  Skill file not found: {skill_file}")
            return

        title = pr.get("title", "")
        content = skill_file.read_text()

        # 1. 如果是新模型，更新表格
        if "new model" in title.lower() or "add model" in title.lower():
            model_info = self.extract_model_info(pr)
            if model_info:
                content = self.update_model_table(content, model_info)
                print(f"  ✅ Updated model table in SKILL.md")

        # 2. 如果是重要特性，添加示例
        if "feature" in title.lower() or "feat" in title.lower():
            example = self.generate_example(pr)
            if example:
                content = self.add_example_section(content, example)
                print(f"  ✅ Added example to SKILL.md")

        # 3. 如果是关键fix，添加troubleshooting
        if any(kw in title.lower() for kw in ["crash", "error", "fix"]):
            note = self.generate_troubleshooting_note(pr)
            if note:
                content = self.add_troubleshooting_note(content, note)
                print(f"  ✅ Added troubleshooting note to SKILL.md")

        # 写回文件
        skill_file.write_text(content)

    def update_updates_log(self, skill_name: str, pr: dict):
        """写入docs/updates/{skill}.md（所有更新）"""

        # 映射skill名称到文件名
        filename_map = {
            "vllm-omni-image-gen": "image-gen",
            "vllm-omni-api": "api",
            "vllm-omni-quantization": "quantization",
            "vllm-omni-perf": "performance",
            "vllm-omni-video-gen": "video-gen",
            "vllm-omni-audio-tts": "audio-tts",
            "vllm-omni-distributed": "distributed",
            "vllm-omni-cicd": "cicd",
        }

        filename = filename_map.get(skill_name, skill_name.replace("vllm-omni-", ""))
        updates_file = self.updates_dir / f"{filename}.md"

        # 如果文件不存在，创建基础结构
        if not updates_file.exists():
            self.create_updates_file(updates_file, skill_name)

        # 生成更新条目
        entry = self.format_update_entry(pr)

        # 读取现有内容
        content = updates_file.read_text()

        # 在"---"后插入新条目
        lines = content.split("\n")
        insert_pos = 0
        for i, line in enumerate(lines):
            if line.strip() == "---" and i > 0:
                insert_pos = i + 1
                break

        # 插入新条目
        lines.insert(insert_pos, entry)

        # 写回文件
        updates_file.write_text("\n".join(lines))
        print(f"  ✅ Written to {updates_file.name}")

    def update_changelog(self, skill_name: str, pr: dict):
        """更新docs/CHANGELOG.md（仅重要更新）"""

        if not self.is_important_update(pr):
            return

        content = self.changelog_file.read_text()

        # 生成changelog条目
        entry = self.format_changelog_entry(skill_name, pr)

        # 在[Unreleased]下的相应section插入
        lines = content.split("\n")
        insert_pos = 0
        section = self.get_changelog_section(pr)

        for i, line in enumerate(lines):
            if f"### {section}" in line:
                insert_pos = i + 1
                break

        if insert_pos > 0:
            lines.insert(insert_pos, entry)
            self.changelog_file.write_text("\n".join(lines))
            print(f"  ✅ Updated CHANGELOG.md")

    def format_update_entry(self, pr: dict) -> str:
        """格式化更新条目（用于docs/updates/{skill}.md）"""
        date = pr["mergedAt"][:10]
        title = re.sub(r'\[.*?\]', '', pr["title"]).strip()

        # 判断类型
        if "feat" in pr["title"].lower() or "add" in pr["title"].lower():
            change_type = "**Added**:"
        elif "fix" in pr["title"].lower():
            change_type = "**Fixed**:"
        else:
            change_type = "**Changed**:"

        # 提取关键信息
        key_info = self.extract_key_info(pr)

        entry = f"""
### {date}
**[PR #{pr['number']}](https://github.com/{self.vllm_omni_repo}/pull/{pr['number']})** - {title}

{change_type}
{key_info}

**Updated in skill**:
- ✅ (auto-marked)

---
"""
        return entry

    def format_changelog_entry(self, skill_name: str, pr: dict) -> str:
        """格式化changelog条目"""
        skill_short = skill_name.replace("vllm-omni-", "")
        title = re.sub(r'\[.*?\]', '', pr["title"]).strip()

        return f"- [{skill_short}] {title} ([#{pr['number']}](https://github.com/{self.vllm_omni_repo}/pull/{pr['number']}))"

    def get_changelog_section(self, pr: dict) -> str:
        """判断应该放在哪个section"""
        title = pr["title"].lower()
        if "feat" in title or "add" in title:
            return "Added"
        elif "fix" in title:
            return "Fixed"
        else:
            return "Changed"

    def extract_key_info(self, pr: dict) -> str:
        """提取关键信息"""
        body = pr.get("body", "")
        title = pr["title"]

        # 简单提取：从body中提取前3行
        lines = [l.strip() for l in body.split("\n") if l.strip() and not l.startswith("#")][:3]

        if lines:
            return "\n".join([f"- {l}" for l in lines])
        else:
            return f"- {re.sub(r'\[.*?\]', '', title).strip()}"

    def extract_model_info(self, pr: dict) -> dict | None:
        """从PR中提取模型信息"""
        # TODO: 实现智能提取
        return None

    def generate_example(self, pr: dict) -> str | None:
        """从PR生成示例代码"""
        # TODO: 实现示例生成
        return None

    def generate_troubleshooting_note(self, pr: dict) -> str | None:
        """生成troubleshooting note"""
        # TODO: 实现troubleshooting note生成
        return None

    def update_model_table(self, content: str, model_info: dict) -> str:
        """更新模型表格"""
        # TODO: 实现表格更新
        return content

    def add_example_section(self, content: str, example: str) -> str:
        """添加示例"""
        # TODO: 实现示例添加
        return content

    def add_troubleshooting_note(self, content: str, note: str) -> str:
        """添加troubleshooting note"""
        # TODO: 实现note添加
        return content

    def create_updates_file(self, updates_file: Path, skill_name: str):
        """Create updates file base structure"""
        skill_display = skill_name.replace("vllm-omni-", "").replace("-", " ").title()
        content = f"""# {skill_name} Update Log

> Last updated: {datetime.now().strftime('%Y-%m-%d')}
> [View all skills updates](../CHANGELOG.md) | [Back to index](README.md)

---

## 2026-03 - Week 1

---

*Maintained by vllm-omni-skills auto-update system*
*Archived every 4 weeks (aligned with vllm-omni release cycle)*
"""
        updates_file.write_text(content)

    def run(self, since_days: int = 7, dry_run: bool = False):
        """执行更新"""
        print(f"🔍 Checking for vllm-omni updates (last {since_days} days)...\n")

        # 获取PRs
        prs = self.get_recent_prs(since_days)
        print(f"📊 Found {len(prs)} recent merged PRs\n")

        if not prs:
            print("✅ No PRs to process")
            return

        # 处理每个PR
        for pr in prs:
            print(f"PR #{pr['number']}: {pr['title']}")

            categories = self.categorize_change(pr)
            if not categories:
                print("  ⚠️  No skill categories detected\n")
                continue

            print(f"  Categories: {', '.join(categories)}")

            for skill_name in categories:
                if not dry_run:
                    # 层1: 更新SKILL.md（仅重要）
                    if self.is_important_update(pr):
                        self.update_skill_core(skill_name, pr)

                    # Layer 2: Write to updates log (all)
                    self.update_updates_log(skill_name, pr)

                    # Layer 3: Update CHANGELOG (important only)
                    self.update_changelog(skill_name, pr)

            print()

        print(f"✅ Processed {len(prs)} PRs")


def main():
    """Main function"""
    import os

    workspace = os.environ.get("WORKSPACE_ROOT", "/Users/hsliu/.openclaw/workspace")
    updater = HybridSkillUpdater(workspace)

    since_days = int(os.environ.get("SINCE_DAYS", "7"))
    dry_run = os.environ.get("DRY_RUN", "false").lower() == "true"

    updater.run(since_days=since_days, dry_run=dry_run)


if __name__ == "__main__":
    main()

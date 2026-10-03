#!/usr/bin/env python3
"""Validate Pi package metadata and SKILL.md frontmatter without extra deps."""
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")

errors = []


def err(msg):
    errors.append(msg)


def read_frontmatter(path):
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    if not text.startswith("---"):
        err(f"{path}: missing opening '---' frontmatter")
        return {}
    end = text.find("\n---", 3)
    if end == -1:
        err(f"{path}: missing closing '---' frontmatter")
        return {}
    block = text[3:end].splitlines()
    out = {}
    i = 0
    while i < len(block):
        line = block[i]
        m = re.match(r"^([A-Za-z_][\w-]*):\s*(.*)$", line)
        if not m:
            i += 1
            continue
        key, val = m.group(1), m.group(2).strip()
        if val in (">", "|", ">-", "|-", ">+", "|+"):
            parts = []
            i += 1
            while i < len(block) and not re.match(r"^[A-Za-z_][\w-]*:", block[i]):
                parts.append(block[i].strip())
                i += 1
            out[key] = " ".join(p for p in parts if p)
            continue
        out[key] = val.strip("'\"")
        i += 1
    return out


def main():
    pkg_path = os.path.join(REPO, "package.json")
    if not os.path.isfile(pkg_path):
        err("package.json not found")
        return report()
    try:
        with open(pkg_path, encoding="utf-8") as fh:
            pkg = json.load(fh)
    except Exception as e:
        err(f"package.json invalid JSON: {e}")
        return report()

    if "pi-package" not in pkg.get("keywords", []):
        err("package.json keywords missing 'pi-package'")

    skill_roots = (pkg.get("pi") or {}).get("skills") or ["./skills"]
    found_any = False
    for root in skill_roots:
        root_abs = os.path.join(REPO, root)
        if not os.path.isdir(root_abs):
            err(f"pi.skills path does not exist: {root}")
            continue
        for entry in sorted(os.listdir(root_abs)):
            d = os.path.join(root_abs, entry)
            skill_md = os.path.join(d, "SKILL.md")
            if not os.path.isfile(skill_md):
                continue
            found_any = True
            fm = read_frontmatter(skill_md)
            name = fm.get("name", "")
            desc = fm.get("description", "")
            if not name:
                err(f"{skill_md}: missing 'name'")
            elif not NAME_RE.match(name) or len(name) > 64:
                err(f"{skill_md}: invalid name '{name}'")
            elif name != entry:
                err(f"{skill_md}: name '{name}' != directory '{entry}'")
            if not desc:
                err(f"{skill_md}: missing 'description'")
            elif len(desc) > 1024:
                err(f"{skill_md}: description too long ({len(desc)} > 1024)")
    if not found_any:
        err("no SKILL.md found under declared pi.skills paths")

    return report()


def report():
    if errors:
        print("Validation failed:")
        for e in errors:
            print("  -", e)
        return 1
    print("Validation OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

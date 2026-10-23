# Claude Code Instructions — CADRE

## Before Writing Any Code

Read `ai/ARCHITECTURE.md` in full, then `ai/PORT_STATUS.md`, `ai/OM4_NEEDS.md` and
`ai/DYMOS4_NEEDS.md`. CADRE is being ported onto om4 and dymos4, whose own architecture
documents (`../om4.git/ai/ARCHITECTURE.md`, `../dymos4.git/ai/ARCHITECTURE.md`) also apply.

When om4 or dymos4 gets in the way, record the gap in the matching `ai/*_NEEDS.md` instead of
working around it, and mark the blocked test `expectedFailure` (ADL-003).

## Environment Setup

This project uses [pixi](https://pixi.sh), with om4 and dymos4 installed editable from the
sibling checkouts `../om4.git` and `../dymos4.git`:

```
pixi install
```

## Common Commands

| Task      | Command         |
|-----------|-----------------|
| Run tests | `pixi run test` |
| Run lint  | `pixi run lint` |

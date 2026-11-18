# Commit rules (ai_elint)

- Conventional Commits: `<type>(optional scope): <description>`.
  Allowed types: build, chore, ci, docs, feat, fix, perf, refactor,
  revert, style, test.
- One simple top-level folder per concern (flat layout). Never nest
  topics under an umbrella folder.
- Never push to master or GitLab unless the user explicitly asks;
  leave work on the chat branch as an MR for review.
- Do not merge to master until the 700 ms NUC E2E target is met.

# Copilot Instructions: Real-Estate-Estimator

**Project:** Real-Estate-Estimator (phData MLE Coding Test)
**Established:** December 7, 2025
**Purpose:** Enforce protocols and maintain context across all AI interactions

---

## MANDATORY: READ BEFORE EVERY RESPONSE

Before responding to ANY user prompt, you MUST:

1. **Consult the Master Protocol:** Read `Generated_Docs/Exploratory/master_protocol.md` in full
2. **Check the Master Log:** Review `logs/master_log.md` for current project status
3. **Apply the Anti-Sampling Directive:** Read every word of every file referenced
4. **Follow the Prime Directive:** All work serves deployment of the model as a scalable REST API
5. **Log Your Work:** Update appropriate logs after each significant action

---

## ANTI-SAMPLING DIRECTIVE

**THIS IS MANDATORY. NON-NEGOTIABLE.**

- Read every file in its ENTIRETY. No sampling. No skimming.
- Do NOT read the first part and infer the rest.
- CSV/data files: Sample headers and first 5-10 rows only (exception)
- Binary files: Note metadata only (exception)
- Files over 10,000 lines: Alert user, get authorization, read in chunks

**Violation Response:** Stop work immediately, re-read comprehensively, re-analyze with full fidelity.

---

## PRIME DIRECTIVE

> **Understand and Deploy the Provided Model as a Scalable, Production-Ready REST API**
>
> Starting from the provided `create_model.py`, achieve complete understanding of every function, every dependency, and every data flow. Fix any issues preventing proper operation. Deploy it as a REST API service that can scale horizontally without stopping. Evaluate its generalization performance honestly. Communicate the solution clearly to both business stakeholders and technical teams.
>
> **Success means:** API running, tested, deployed in Docker, model understood, performance documented, presentation ready, code on GitHub.

---

## SUB-PROTOCOLS INDEX

| Sub-Protocol | Purpose | Location |
|--------------|---------|----------|
| Analysis | Analyze findings 100% comprehensively | Master Protocol Section 3.1 |
| Research | Identify and use tools/references needed | Master Protocol Section 3.2 |
| Generation | Code standards, NO emojis/special chars | Master Protocol Section 3.3 |
| Logging | Log everything, update after each action | Master Protocol Section 3.4 |
| Tool Identification | Find best tools/MCPs for tasks | Master Protocol Section 3.5 |
| Reference Material | Find best docs/SDKs/guides | Master Protocol Section 3.6 |
| Tool/Reference Request | Acquire missing resources | Master Protocol Section 3.7 |
| Evaluation | Comprehensive model/API/data evaluation | Master Protocol Section 3.8 |

---

## GENERATION SUB-PROTOCOL RULES

**STRICTLY ENFORCED:**

- NO emojis in code or documentation
- NO special characters unless expressly necessary AND authorized
- Clean, readable code with proper comments
- Type hints in Python
- Docstrings for all functions
- Error handling implemented
- Logging statements where appropriate

---

## LOGGING SUB-PROTOCOL RULES

**After EVERY significant action:**

1. Update the appropriate date-stamped log in `logs/`
2. Update `logs/master_log.md` if phase status changes
3. Use standard format:

```
## Task: [Clear task name]
**Timestamp:** YYYY-MM-DD HH:MM UTC
**Status:** [In Progress | Completed | Blocked | On Hold]
**Details:**
- What was attempted
- What was discovered
- Key decisions made
- Blockers encountered

**Next Step:** [What happens next]

---
```

**Log Files:**
- `logs/master_log.md` - Central index of all logs
- `logs/YYYY-MM-DD_analysis_log.md` - Analysis work
- `logs/YYYY-MM-DD_api_implementation_log.md` - API development
- `logs/YYYY-MM-DD_issues_and_resolutions_log.md` - Bugs and fixes
- `logs/YYYY-MM-DD_model_evaluation_log.md` - Evaluation results

---

## KEY PROJECT FILES

```
Protocols & Logs:
- .github/copilot-instructions.md      [THIS FILE - entry point]
- Generated_Docs/Exploratory/master_protocol.md  [Full protocol details]
- logs/master_log.md                   [Project status index]

Reference Materials:
- Reference_Docs/mle-project-challenge-2/README.md    [phData requirements]
- Reference_Docs/mle-project-challenge-2/create_model.py [Original code, BUG on line 14]
- Generated_Docs/Exploratory/PROJECT_ANALYSIS.md      [Full analysis]

Source Code:
- src/train.py      [Training script with bug fix and MLflow]
- src/evaluate.py   [Evaluation script]
- src/main.py       [FastAPI application - TO BE CREATED]

Data:
- data/kc_house_data.csv           [21,613 training samples]
- data/zipcode_demographics.csv    [83 zipcodes, 27 features]
- data/future_unseen_examples.csv  [300 test examples]
```

---

## DECISION FRAMEWORK

When faced with choices, use this hierarchy:

1. **phData Requirements** - What they explicitly asked for (must do)
2. **Best Practices** - Industry standards for MLOps/API development
3. **Efficiency** - Simplest solution that meets requirements
4. **Extensibility** - Can it be improved later without major refactor?
5. **Learning** - Opportunity to demonstrate technical capability

---

## RESTART PROTOCOL

If context window is getting full or at session end:

1. Review entire chat history
2. Ensure all work is logged to appropriate files
3. Update master_log.md with current status
4. Consult master protocol for where we are vs. plan
5. Prepare comprehensive restart prompt with:
   - Current phase and status
   - Files to read for context
   - Immediate next actions
   - Any blockers or decisions needed

---

## COMMUNICATION STANDARDS

- Be direct and clear
- State status before explaining details
- Provide context for decisions
- Flag blockers immediately
- Show progress transparently
- Code comments: Clear English, explain WHY not just WHAT

---

## CURRENT PROJECT STATUS

**Phase:** Implementation (Phase B - Training Scripts)
**Branch:** develop
**Last Action:** Created src/train.py and src/evaluate.py
**Next Action:** Update logs, then continue to Phase C (FastAPI)

---

## CHECKLIST: BEFORE RESPONDING

- [ ] Have I read the master protocol?
- [ ] Have I checked the master log for current status?
- [ ] Am I following the anti-sampling directive?
- [ ] Does my response serve the prime directive?
- [ ] Have I avoided emojis and special characters?
- [ ] Will I update the appropriate logs after this action?

---

**Last Updated:** 2025-12-07
**Maintained By:** AI Assistant following established protocols

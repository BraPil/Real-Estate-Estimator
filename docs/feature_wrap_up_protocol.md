# Feature Wrap-Up Protocol

**Purpose:** Standardized checklist for completing a feature version and transitioning to the next.

---

## Pre-Wrap-Up Checklist

### 1. Verify Feature Completion
- [ ] All planned functionality implemented
- [ ] All tests passing
- [ ] API endpoints working correctly
- [ ] Model metrics validated with `evaluate.py`
- [ ] No critical bugs or regressions

### 2. Model Verification
```powershell
# Verify correct model is deployed
python -c "import pickle; m = pickle.load(open('model/model.pkl', 'rb')); print(f'Model: {m.named_steps}')"

# Run evaluation
python src/evaluate.py --test-size 0.2

# Test API (ensure server is running)
python scripts/test_api.py
```

---

## Documentation Updates

### 3. Update Core Documentation

#### metrics.json
Update `model/metrics.json` with:
- New version number
- Model type and hyperparameters
- Test metrics (R², MAE, RMSE)
- Version history comparison

#### Version Roadmap
Update `docs/V2_Detailed_Roadmap.md`:
- Mark [Current Version] as ✅ **COMPLETE**
- Update [Next Version] status to 🔜 **NEXT**
- Add decision log entry for [Current Version]

#### RESTART Document
Update `RESTART_[DATE].md`:
- Current branch: `feature/[Next Version]-[feature-name]`
- Last completed: [Current Version]
- Model configuration for [Current Version]
- Performance history table

### 4. Create Version Completion Summary
Create `docs/[Current Version]_Completion_Summary.md`:
- Executive summary
- Performance comparison table
- Technical details
- Files created/modified
- Lessons learned
- Next steps

### 5. Create Educational Document (Optional)
Create `docs/[Current Version]_Deep_Dive.md`:
- What was accomplished
- Technical concepts explained
- Code walkthrough
- Key decisions and rationale

### 6. Document Changes
Create `docs/Changes_from_[Previous Version]_to_[Current Version]_[DATE].md`:
- Files added
- Files modified
- Configuration changes
- Dependencies added
- Breaking changes (if any)

### 7. Update Correction Log
Update `logs/human_in_the_loop_corrections.md`:
- Add any corrections made during development
- Document debugging steps
- Note any gotchas discovered

---

## Git Operations

### 8. Stage and Review Changes
```powershell
# Check status
git status

# Review all changes
git diff

# Add all changes
git add -A
```

### 9. Commit with Descriptive Message
```powershell
git commit -m "[Current Version]: [Brief description of changes]

- [Key change 1]
- [Key change 2]
- [Key change 3]

Performance: MAE [old] -> [new] ([improvement]%)"
```

### 10. Push Feature Branch
```powershell
git push origin feature/[Current Version]-[feature-name]
```

### 11. Merge to Develop
```powershell
# Switch to develop
git checkout develop

# Merge feature branch
git merge feature/[Current Version]-[feature-name]

# Push develop
git push origin develop
```

### 12. Merge to Main (Release)
```powershell
# Switch to main
git checkout main

# Merge develop
git merge develop

# Tag release
git tag -a [Current Version] -m "[Current Version]: [Description]"

# Push main and tags
git push origin main --tags
```

### 13. Create Next Version Branch
```powershell
# Create and switch to new feature branch
git checkout -b feature/[Next Version]-[feature-name]

# Push new branch
git push -u origin feature/[Next Version]-[feature-name]
```

---

## Post-Wrap-Up Verification

### 14. Verify Git State
```powershell
# Verify branches
git branch -a

# Verify tags
git tag -l

# Verify remote
git log --oneline -5 origin/main
```

### 15. Verify Deployment
- [ ] API serving correct model version
- [ ] Health endpoint shows correct version
- [ ] All endpoints functional

---

## Template Variables

Replace these placeholders throughout:
- `[Previous Version]` - e.g., v2.3
- `[Current Version]` - e.g., v2.4
- `[Next Version]` - e.g., v2.5
- `[DATE]` - e.g., 20251208
- `[feature-name]` - e.g., model-alternatives

---

**Protocol Version:** 1.0  
**Last Updated:** 2025-12-08

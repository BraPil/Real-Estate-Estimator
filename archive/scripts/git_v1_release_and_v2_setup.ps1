# Git V1 Release and V2 Setup Script
# Run this from the project root: .\scripts\git_v1_release_and_v2_setup.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " V1 RELEASE AND V2 BRANCH SETUP" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Step 1: Show current status
Write-Host "`n[STEP 1] Current Git Status:" -ForegroundColor Yellow
git status
git branch -a

# Step 2: Stage all V1 files
Write-Host "`n[STEP 2] Staging all changes..." -ForegroundColor Yellow
git add .
git status --short

# Step 3: Commit V1
Write-Host "`n[STEP 3] Committing V1..." -ForegroundColor Yellow
$commitMessage = @"
feat: V1 complete - REST API with ML prediction

V1 Implementation includes:
- REST API (FastAPI) with /predict and /predict-minimal endpoints
- Accepts all 18 columns from future_unseen_examples.csv
- Demographics enrichment on backend (26 features from 70 zipcodes)
- MLflow integration for experiment tracking
- Docker support (Dockerfile, docker-compose.yml)
- Data vintage warnings in all responses
- Comprehensive documentation

Bug fix: Corrected DEMOGRAPHICS_PATH from kc_house_data.csv to zipcode_demographics.csv

Test results: 3/3 predictions successful, 3/3 minimal predictions successful
"@

git commit -m $commitMessage

# Step 4: Tag V1 release
Write-Host "`n[STEP 4] Tagging V1.0.0 release..." -ForegroundColor Yellow
git tag -a v1.0.0 -m "V1.0.0 - Production-ready REST API for home price prediction"

# Step 5: Create V2 feature branch
Write-Host "`n[STEP 5] Creating feature/v2-model-improvements branch..." -ForegroundColor Yellow
git checkout -b feature/v2-model-improvements

# Step 6: Show final state
Write-Host "`n[STEP 6] Final State:" -ForegroundColor Yellow
Write-Host "Current branch:" -ForegroundColor Green
git branch --show-current
Write-Host "`nAll branches:" -ForegroundColor Green
git branch -a
Write-Host "`nTags:" -ForegroundColor Green
git tag -l

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host " SETUP COMPLETE!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host @"

Next steps:
1. Push main branch:     git push origin main
2. Push V1 tag:          git push origin v1.0.0
3. Push feature branch:  git push -u origin feature/v2-model-improvements

V2 Development can now begin on the feature/v2-model-improvements branch.
"@ -ForegroundColor White

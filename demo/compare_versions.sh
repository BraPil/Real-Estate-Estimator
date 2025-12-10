#!/bin/bash
# Demo Script: Compare V1 vs V2.5 vs V3.3 APIs
# Run this after: docker-compose -f docker-compose.demo.yml up --build
#
# This script demonstrates the evolution from V1 MVP to V3.3 Production

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Sample house data (all 18 columns)
HOUSE_DATA='{
  "bedrooms": 3,
  "bathrooms": 2.5,
  "sqft_living": 2000,
  "sqft_lot": 5000,
  "floors": 2,
  "waterfront": 0,
  "view": 0,
  "condition": 3,
  "grade": 8,
  "sqft_above": 1500,
  "sqft_basement": 500,
  "yr_built": 2010,
  "yr_renovated": 0,
  "zipcode": "98103",
  "lat": 47.6,
  "long": -122.3,
  "sqft_living15": 1900,
  "sqft_lot15": 4800
}'

# Minimal house data (for predict-minimal endpoint - needs 11 fields)
MINIMAL_DATA='{
  "bedrooms": 3,
  "bathrooms": 2.5,
  "sqft_living": 2000,
  "sqft_lot": 5000,
  "floors": 2,
  "grade": 8,
  "condition": 3,
  "sqft_above": 1500,
  "sqft_basement": 500,
  "yr_built": 2010,
  "zipcode": "98103"
}'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Real Estate API Version Comparison   ${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if services are running
echo -e "${YELLOW}Checking service availability...${NC}"
echo ""

# V1 Health Check
echo -e "${CYAN}[V1 MVP] Health Check (Port 8000)${NC}"
echo -e "Endpoint: GET /health  (NO /api/v1 prefix)"
curl -s http://localhost:8000/health | python3 -m json.tool 2>/dev/null || echo "V1 not running"
echo ""

# V2.5 Health Check
echo -e "${GREEN}[V2.5] Health Check (Port 8001)${NC}"
echo -e "Endpoint: GET /api/v1/health"
curl -s http://localhost:8001/api/v1/health | python3 -m json.tool 2>/dev/null || echo "V2.5 not running"
echo ""

# V3.3 Health Check
echo -e "${GREEN}[V3.3] Health Check (Port 8002)${NC}"
echo -e "Endpoint: GET /api/v1/health"
curl -s http://localhost:8002/api/v1/health | python3 -m json.tool 2>/dev/null || echo "V3.3 not running"
echo ""

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Prediction Comparison (Same Input)   ${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# V1 Predict
echo -e "${CYAN}[V1 MVP] Prediction (7 features used)${NC}"
echo -e "Endpoint: POST /predict  (NO /api/v1 prefix)"
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d "$HOUSE_DATA" | python3 -m json.tool 2>/dev/null || echo "Failed"
echo ""

# V2.5 Full Predict
echo -e "${GREEN}[V2.5] Full Prediction${NC}"
echo -e "Endpoint: POST /api/v1/predict"
curl -s -X POST http://localhost:8001/api/v1/predict \
  -H "Content-Type: application/json" \
  -d "$HOUSE_DATA" | python3 -m json.tool 2>/dev/null || echo "Failed"
echo ""

# V3.3 Full Predict
echo -e "${GREEN}[V3.3] Full Prediction${NC}"
echo -e "Endpoint: POST /api/v1/predict"
curl -s -X POST http://localhost:8002/api/v1/predict \
  -H "Content-Type: application/json" \
  -d "$HOUSE_DATA" | python3 -m json.tool 2>/dev/null || echo "Failed"
echo ""

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  V2.5 Experimental Endpoints          ${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# V2.5 Minimal Predict
echo -e "${GREEN}[V2.5] Minimal Prediction (7 features only)${NC}"
echo -e "Endpoint: POST /api/v1/predict-minimal"
curl -s -X POST http://localhost:8001/api/v1/predict-minimal \
  -H "Content-Type: application/json" \
  -d "$MINIMAL_DATA" | python3 -m json.tool 2>/dev/null || echo "Failed"
echo ""

# V2.5 Adaptive Predict
echo -e "${GREEN}[V2.5] Adaptive Prediction (tier-based routing)${NC}"
echo -e "Endpoint: POST /api/v1/predict-adaptive"
curl -s -X POST http://localhost:8001/api/v1/predict-adaptive \
  -H "Content-Type: application/json" \
  -d "$HOUSE_DATA" | python3 -m json.tool 2>/dev/null || echo "Failed"
echo ""

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  API Documentation Links              ${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "V1 MVP Swagger UI:  ${YELLOW}http://localhost:8000/docs${NC}"
echo -e "V2.5 Swagger UI:    ${YELLOW}http://localhost:8001/docs${NC}"
echo -e "V3.3 Swagger UI:    ${YELLOW}http://localhost:8002/docs${NC}"
echo ""

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Key Differences Summary              ${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${CYAN}V1 MVP:${NC}"
echo -e "  - Endpoints: /health, /predict (no /api/v1 prefix)"
echo -e "  - Features: 7 home + 26 demographic = 33 total"
echo -e "  - Data: 2014-2015 vintage"
echo ""
echo -e "${GREEN}V2.5:${NC}"
echo -e "  - Endpoints: /api/v1/predict, /predict-minimal, /predict-adaptive"
echo -e "  - Features: 17 home + 26 demographic = 43 total"
echo -e "  - Data: 2014-2015 vintage"
echo -e "  - Added: Tier-based adaptive routing experiment"
echo ""
echo -e "${GREEN}V3.3:${NC}"
echo -e "  - Endpoints: Same as V2.5"
echo -e "  - Features: 17 home + 26 demographic = 43 total"
echo -e "  - Data: 2020-2024 vintage (fresh data)"
echo -e "  - Added: MLflow integration, production hardening"
echo ""
echo -e "${GREEN}Demo complete!${NC}"

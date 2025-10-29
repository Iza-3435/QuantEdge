#!/bin/bash

# AI MARKET INTELLIGENCE PLATFORM
# Single entry point launcher

clear

# Check if Python is available
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "Error: Python is not installed or not in PATH"
    exit 1
fi

# Use python3 if available, otherwise python
PYTHON_CMD=$(command -v python3 || command -v python)

# Color codes - matching Rich gray theme
GRAY='\033[38;5;244m'
WHITE='\033[1;37m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Banner - matching MAIN_MENU.py theme
echo -e "${GRAY}┌───────────────────────────────────────────────────────────────────────────────┐${NC}"
echo -e "${GRAY}│                                                                               │${NC}"
echo -e "${GRAY}│${WHITE}  AI MARKET INTELLIGENCE SYSTEM                                                ${GRAY}│${NC}"
echo -e "${GRAY}│${WHITE}  Production-Grade Investment Platform                                        ${GRAY}│${NC}"
echo -e "${GRAY}│                                                                               │${NC}"
echo -e "${GRAY}└───────────────────────────────────────────────────────────────────────────────┘${NC}"
echo ""

# Launch the main menu
$PYTHON_CMD MAIN_MENU.py

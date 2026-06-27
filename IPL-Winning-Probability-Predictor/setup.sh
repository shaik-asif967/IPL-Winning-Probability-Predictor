#!/bin/bash
mkdir -p ~/.streamlit/

echo "[server]
headless = true
port = 3000
enableCORS = false

[theme]
primaryColor = '#10b981'
backgroundColor = '#0b0f19'
secondaryBackgroundColor = '#0f172a'
textColor = '#f1f5f9'
font = 'sans serif'
" > ~/.streamlit/config.toml

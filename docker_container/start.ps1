$ErrorActionPreference = "Stop"

# Prompt for Qdrant API key (required)
$qdrantApiKey = Read-Host -Prompt "Enter Qdrant API Key (required)"
if ([string]::IsNullOrWhiteSpace($qdrantApiKey)) { Write-Error "QDRANT_API_KEY is required."; exit 1 }

# Write .env for compose
$envPath = Join-Path $PSScriptRoot ".env"
"QDRANT_API_KEY=$qdrantApiKey" | Out-File -FilePath $envPath -Encoding UTF8 -Force
Write-Host "Created $envPath with QDRANT_API_KEY."

Write-Host "Starting containers..."
Push-Location $PSScriptRoot
try {
  docker volume create ai-assistant-qdrant | Out-Null
  docker compose up -d
}
finally {
  Pop-Location
}

Write-Host "Done. Access Qdrant via http://localhost:6333 (API key required)."


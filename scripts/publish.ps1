# Create the GitHub repo and push. Run after `gh auth login`.
#
#   gh auth login          # once, interactive (GitHub.com -> HTTPS -> browser)
#   scripts\publish.ps1 -Name RecycleLM -Private
param(
    [string]$Name = "RecycleLM",
    [switch]$Private
)
$ErrorActionPreference = "Stop"
$gh = "C:\Program Files\GitHub CLI\gh.exe"
if (-not (Test-Path $gh)) { $gh = "gh" }

& $gh auth status *> $null
if ($LASTEXITCODE -ne 0) { throw "not authenticated: run 'gh auth login' first" }

Push-Location (Join-Path $PSScriptRoot "..")
try {
    $vis = if ($Private) { "--private" } else { "--public" }
    & $gh repo create $Name $vis --source . --remote origin --push
} finally {
    Pop-Location
}

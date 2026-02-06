param(
  [string] = (Split-Path -Parent System.Management.Automation.InvocationInfo.MyCommand.Path),
  [string] = "C:\Users\admin\.codex\skills"
)

 = Join-Path  'codex_skills'
if (-not (Test-Path )) {
  throw "Missing "
}

New-Item -ItemType Directory -Force -Path  | Out-Null

Get-ChildItem -Path  -Directory | ForEach-Object {
   = .Name
  if ( -eq '.system') { return }

   = Join-Path  
  New-Item -ItemType Directory -Force -Path  | Out-Null

  Copy-Item -Force -Recurse -Path (Join-Path .FullName '*') -Destination 
}

Write-Host "Installed skills into "

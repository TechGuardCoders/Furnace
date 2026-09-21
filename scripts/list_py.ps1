Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -notmatch 'hermes_cli' } |
  Select-Object ProcessId, @{n='Cmd';e={$_.CommandLine.Substring(0,[Math]::Min(140,$_.CommandLine.Length))}} |
  Format-List
Write-Output "--- ports 3100/3200 ---"
(Get-NetTCPConnection -State Listen -LocalPort 3100 -ErrorAction SilentlyContinue).OwningProcess
(Get-NetTCPConnection -State Listen -LocalPort 3200 -ErrorAction SilentlyContinue).OwningProcess

# Export this lesson's DOCX files to review/rendered/*.pdf with Microsoft Word.
# Each file runs in its own job with a timeout, so a Word dialog cannot hang the whole batch.
param([switch]$IncludeReleases,[int]$TimeoutSeconds=150)
$ErrorActionPreference = 'Stop'
$lessonRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$renderRoot = Join-Path $lessonRoot 'review\rendered'
New-Item -ItemType Directory -Force -Path $renderRoot | Out-Null
$documents = @(Get-ChildItem -LiteralPath $lessonRoot -Filter 'pbl-svm-margin-*.docx')
if ($IncludeReleases) { $documents += @(Get-ChildItem -LiteralPath (Join-Path $lessonRoot 'student-release') -Filter '*.docx') }
$export = {
    param($in,$pdf)
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false; $word.DisplayAlerts = 0
    try {
        $doc = $word.Documents.Open($in, $false, $true, $false)
        $doc.ExportAsFixedFormat($pdf, 17)
        $pages = $doc.ComputeStatistics(2); $doc.Close(0); $pages
    } finally { $word.Quit() }
}
foreach ($file in $documents) {
    $pdf = Join-Path $renderRoot ($file.BaseName + '.pdf')
    $job = Start-Job -ScriptBlock $export -ArgumentList $file.FullName,$pdf
    if (Wait-Job $job -Timeout $TimeoutSeconds) { Write-Output ($file.Name + ' -> ' + (Receive-Job $job) + ' pages') }
    else {
        Write-Output ($file.Name + ' -> TIMEOUT'); Stop-Job $job
        Get-CimInstance Win32_Process -Filter "Name='WINWORD.EXE'" | Where-Object { $_.CommandLine -match 'Automation' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
    }
    Remove-Job $job -Force
}

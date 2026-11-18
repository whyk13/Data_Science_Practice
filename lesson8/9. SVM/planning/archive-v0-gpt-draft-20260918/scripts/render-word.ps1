param([switch]$IncludeReleases)
$ErrorActionPreference = 'Stop'
$lessonRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$renderRoot = Join-Path $lessonRoot 'review\rendered'
New-Item -ItemType Directory -Force -Path $renderRoot | Out-Null
$documents = @(Get-ChildItem -LiteralPath $lessonRoot -Filter 'pbl-svm-margin-*.docx')
if ($IncludeReleases) { $documents += @(Get-ChildItem -LiteralPath (Join-Path $lessonRoot 'student-release') -Filter '*.docx') }
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    foreach ($file in $documents) {
        $doc = $null
        try {
            $doc = $word.Documents.Open($file.FullName, $false, $true)
            $pdf = Join-Path $renderRoot ($file.BaseName + '.pdf')
            $doc.ExportAsFixedFormat($pdf, 17)
            Write-Output ($file.Name + ' -> ' + $doc.ComputeStatistics(2) + ' pages')
        } finally { if ($null -ne $doc) { $doc.Close(0) } }
    }
} finally { $word.Quit() }

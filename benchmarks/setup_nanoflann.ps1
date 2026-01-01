# PowerShell script to download nanoflann.hpp

Write-Host "Downloading nanoflann.hpp..." -ForegroundColor Green

$url = "https://raw.githubusercontent.com/jlblancoc/nanoflann/master/include/nanoflann.hpp"
$outputPath = "include\nanoflann.hpp"

try {
    Invoke-WebRequest -Uri $url -OutFile $outputPath
    Write-Host "✓ nanoflann.hpp successfully downloaded to benchmarks\include\" -ForegroundColor Green
    Write-Host "You can now build the benchmark suite" -ForegroundColor Cyan
}
catch {
    Write-Host "✗ Failed to download nanoflann.hpp" -ForegroundColor Red
    Write-Host "Error: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please manually download from:" -ForegroundColor Yellow
    Write-Host $url -ForegroundColor Yellow
    Write-Host "and place it in benchmarks\include\" -ForegroundColor Yellow
    exit 1
}

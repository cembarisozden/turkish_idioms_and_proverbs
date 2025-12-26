# Install dependencies for Turkish NLP Pipeline
# Run this script in PowerShell: .\install_dependencies.ps1

Write-Host "Installing required packages..." -ForegroundColor Green

# Try different methods to install packages
$methods = @(
    { python -m pip install --upgrade pip },
    { py -m pip install --upgrade pip },
    { python3 -m pip install --upgrade pip }
)

$pipFound = $false
foreach ($method in $methods) {
    try {
        & $method
        $pipFound = $true
        break
    } catch {
        continue
    }
}

if (-not $pipFound) {
    Write-Host "pip not found. Please install pip first:" -ForegroundColor Yellow
    Write-Host "1. Download get-pip.py from https://bootstrap.pypa.io/get-pip.py" -ForegroundColor Yellow
    Write-Host "2. Run: python get-pip.py" -ForegroundColor Yellow
    exit 1
}

# Install packages
$packages = @("torch", "transformers", "scikit-learn", "pandas", "numpy", "openpyxl")

foreach ($package in $packages) {
    Write-Host "Installing $package..." -ForegroundColor Cyan
    try {
        python -m pip install $package
    } catch {
        try {
            py -m pip install $package
        } catch {
            Write-Host "Failed to install $package" -ForegroundColor Red
        }
    }
}

Write-Host "`nInstallation complete!" -ForegroundColor Green
Write-Host "You can now run: python scripts/run_prepare_data.py" -ForegroundColor Green



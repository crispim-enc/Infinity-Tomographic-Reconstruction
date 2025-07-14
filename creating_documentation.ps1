# PowerShell Script: creating_documentation.ps1

# Clear previous HTML build pages except 'source'
Write-Host "Cleaning previous HTML build..." -ForegroundColor Cyan
Get-ChildItem -Path .\docs\ -Exclude 'source' | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

# Clear sphinx-apidoc generated files but keep important files/folders
Write-Host "Cleaning old sphinx-apidoc files..." -ForegroundColor Cyan
$keepFolders = @('_static', '_templates', 'examples', 'images')
$keepFiles = @('conf.py', 'custom.css', 'index.rst', 'installation.rst', 'modules.rst', 'references.rst', 'userguide.rst','.nojekyll')

# Delete folders not in the keep list
Get-ChildItem -Path .\docs\source\ -Directory | Where-Object { $_.Name -notin $keepFolders } | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

# Delete files not in the keep list
Get-ChildItem -Path .\docs\source\ -File | Where-Object { $_.Name -notin $keepFiles } | Remove-Item -Force -ErrorAction SilentlyContinue

# Generate updated API documentation
Write-Host "Running sphinx-apidoc..." -ForegroundColor Cyan
sphinx-apidoc -f -o docs/source/ src/toor

# Change the header in modules.rst
$modulesPath = ".\docs\source\modules.rst"
$content = Get-Content $modulesPath
if ($content.Length -ge 2) {
    $content[0] = "API reference"
    $content[1] = "=" * $content[0].Length
    $content | Set-Content $modulesPath
}


# Build the documentation
Write-Host "Building HTML docs..." -ForegroundColor Cyan
sphinx-build -b html docs/source docs/ -n -v

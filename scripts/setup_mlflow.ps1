
$MlflowDir = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\mlflow_data"))
New-Item -ItemType Directory -Force -Path $MlflowDir | Out-Null

$DbPath = (Join-Path $MlflowDir "mlflow.db").Replace("\", "/")
$ArtifactPath = (Join-Path $MlflowDir "artifacts").Replace("\", "/")
$BackendUri = "sqlite:///$DbPath"
$ArtifactUri = "file:///$ArtifactPath"

Write-Host ">>> Starting MLflow Tracking Server at http://localhost:5000 ..."

mlflow server `
  --backend-store-uri $BackendUri `
  --default-artifact-root $ArtifactUri `
  --host 0.0.0.0 `
  --port 5000

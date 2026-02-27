# Virtual Environment Setup

## ✅ Virtual Environment Created

Location: `.\venv\`

## Activation Instructions

### Windows PowerShell
```powershell
.\venv\Scripts\Activate.ps1
```

### Windows Command Prompt
```cmd
.\venv\Scripts\activate.bat
```

### Linux/macOS
```bash
source venv/bin/activate
```

## Installing Dependencies

After activation, install all packages:
```powershell
pip install -r requirements.txt
```

## Deactivation

To exit the virtual environment:
```powershell
deactivate
```

## Usage

### 1. Activate the environment
```powershell
.\venv\Scripts\Activate.ps1
```

### 2. Run your application
```powershell
# Run the API server
python app.py

# Or run the quickstart example
python quickstart.py
```

### 3. Verify setup
```powershell
python -c "from src.pipeline import RAGPipeline; print('✓ Everything works!')"
```

## VSCode Integration

VSCode should automatically detect the virtual environment. If not:

1. Press `Ctrl+Shift+P`
2. Type "Python: Select Interpreter"
3. Choose `.\venv\Scripts\python.exe`

## Benefits of Virtual Environment

- ✅ Isolated dependencies (won't affect system Python)
- ✅ Reproducible environment
- ✅ Easy to delete and recreate
- ✅ Multiple Python projects can coexist

## Troubleshooting

### PowerShell Execution Policy Error
If you get an execution policy error, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Dependencies Already Installed
If packages were installed globally, they need to be reinstalled in the venv:
```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

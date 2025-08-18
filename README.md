# FNS Pareto Analyzer

A desktop tool for exploring Pareto-optimal assignments of tasks to inspectors. The application uses an evolutionary optimisation pipeline and a PySide6 GUI to visualise results.

## Features
- Optimises task distribution using DEAP and displays the Pareto front in an interactive plot.
- Visualises efficiency forecasts and workload comparisons.

## Project structure
```
.
├── app/                # GUI application and optimisation logic
├── data/               # Example CSV datasets
├── requirements.txt    # Python dependencies
```
See the `architecture` file for a detailed breakdown of modules and roles.

## Installation
1. Create and activate a virtual environment (optional).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the GUI application from the repository root:
```bash
python app/main.py
```
The application will load the sample data from the `data/` directory and display optimisation results.

## Building a standalone executable
Use PyInstaller to bundle the application:
```bash
pyinstaller --noconfirm --clean --name ParetoAnalyzer --windowed --collect-data matplotlib --collect-submodules matplotlib --collect-submodules matplotlib.backends --collect-data PySide6 --collect-binaries PySide6 --exclude-module PyQt5 --exclude-module PyQt5.sip --exclude-module PyQt6 --exclude-module PySide2 --exclude-module qtpy --add-data "app/style;style" app/main.py
```
This command produces a platform-specific executable in the `dist/` directory.

## License
This project is provided as-is.

# app/main.py
from __future__ import annotations
import sys
from PySide6.QtWidgets import QApplication
from style.colors import STYLE_FNS
from main_window import MainWindow

def main() -> int:
    app = QApplication(sys.argv)
    app.setStyleSheet(
        f"""
        QPushButton {{ background-color: {STYLE_FNS}; color: white; padding: 4px 8px; border: none; border-radius: 4px; }}
        QPushButton:hover {{ background-color: {STYLE_FNS}; }}
        QGroupBox {{ border: 1px solid {STYLE_FNS}; margin-top: 6px; }}
        QGroupBox:title {{ subcontrol-origin: margin; left: 10px; padding: 0 3px 0 3px; }}
        """
    )
    window = MainWindow()
    window.show()
    return app.exec()

if __name__ == "__main__":
    raise SystemExit(main())

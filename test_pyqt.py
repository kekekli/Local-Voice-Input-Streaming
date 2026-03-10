import sys
from PyQt6.QtWidgets import QApplication, QMainWindow

app = QApplication(sys.argv)
window = QMainWindow()
window.show()
print("PyQt started")
import time
time.sleep(1)
sys.exit(0)

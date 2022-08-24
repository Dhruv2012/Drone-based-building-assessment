import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget, QComboBox, QPushButton, QFileDialog, QVBoxLayout

class MyApp(QWidget):
    def __init__(self):
        super().__init__()

    def getDirectory(self):
        # response = QFileDialog.getExistingDirectory(self, str("Open Directory"))
        response = QFileDialog.getExistingDirectory(
            self,
            caption='Select a folder'
        )
        return response 

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet('''
        QWidget {
            font-size: 35px;
        }
    ''')
    
    myApp = MyApp()
    path = myApp.getDirectory()
    print(path)
    # myApp.show()

    try:
        sys.exit() 
    except SystemExit:
        pass
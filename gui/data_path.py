import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget, QComboBox, QPushButton, QFileDialog, QVBoxLayout

filepath = ""
mode = ""

class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.window_width, self.window_height = 800, 200
        self.setMinimumSize(self.window_width, self.window_height)

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.options = ('Distance module: From Top', 'Distance module: In Between', 'Distance module: Frontal', 'Roof area calculation', 'Roof Object detection')  

        self.combo = QComboBox()
        self.combo.addItems(self.options)
        layout.addWidget(self.combo)

        btn = QPushButton('Video Path')
        btn.clicked.connect(self.getFileName)
        layout.addWidget(btn)

        btn = QPushButton('Submit')
        btn.clicked.connect(self.submitDialog)
        layout.addWidget(btn)

    def submitDialog(self):
        global mode
        option = self.options.index(self.combo.currentText())
        mode = self.options[option]
        self.close()
        # print(mode)

    def getFileName(self):
        global filepath
        response = QFileDialog.getOpenFileName(self, str("Select a video ile"),
                                       os.getcwd(),
                                       str("Videos (*.mp4 *.mov)"))
        filepath = response

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet('''
        QWidget {
            font-size: 35px;
        }
    ''')
    
    myApp = MyApp()
    myApp.show()
    try:
        sys.exit(app.exec_())
    except SystemExit:
        print(mode)
        print(filepath)
        print()
        print('Closing Window...')
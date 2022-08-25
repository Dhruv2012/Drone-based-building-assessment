from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import sys
import cv2
import os
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt,QObject
import glob

class DisplayResults(QMainWindow):
    def __init__(self):
        super(DisplayResults, self).__init__()
        
        self.setWindowTitle("Displaying Intermediate Results")
        # self.window_width, self.window_height = 1800, 600
        # self.setMinimumSize(self.window_width, self.window_height)
        self.setWindowState(Qt.WindowMaximized)
        
        self.main_layout = QVBoxLayout()

        self.roof_area_photo = QtWidgets.QLabel()
        self.roof_area_photo.setPixmap(QtGui.QPixmap("./gui_images/Roof_Area_Calculation.png"))
        self.roof_area_photo.setScaledContents(True)
        self.main_layout.addWidget(self.roof_area_photo)


        self.buttons = QHBoxLayout()

        self.intermediate_button = QPushButton('Show Intermediate Results')
        self.intermediate_button.clicked.connect(self.show_intermediate_results)
        self.buttons.addWidget(self.intermediate_button)
        
        self.final_button = QPushButton('Show Final Results')
        self.final_button.clicked.connect(self.show_final_results)
        self.buttons.addWidget(self.final_button)

        
        self.main_layout.addLayout(self.buttons)
        self.main_layout.setSpacing(200)

        widget = QWidget()
        widget.setLayout(self.main_layout)
        self.setCentralWidget(widget)

    def retranslateUi(self, DisplayResults):
        _translate = QtCore.QCoreApplication.translate
        DisplayResults.setWindowTitle(_translate("Displaying Intermediate Results", "Displaying Intermediate Results"))
        self.final_button.setText(_translate("Displaying Intermediate Results", "Skip to Final Results"))
        self.intermediate_button.setText(_translate("Displaying Intermediate Results", "Show Intermediate Results"))

    def show_intermediate_results(self):
        folder_path = './test_folder'
        latest_image = max(glob.iglob(folder_path + '/*'), key=os.path.getctime)
        print(latest_image)
        if latest_image == './test_folder/done.txt':
            _translate = QtCore.QCoreApplication.translate
            self.roof_area_photo.setPixmap(QtGui.QPixmap("./gui_images/Roof_Area_Calculation.png"))
            self.intermediate_button.setText(_translate("Assessment of Civil Structures", "All Intermediate Results Displayed. Press Again to Quit."))
            self.intermediate_button.clicked.connect(DisplayResults.close)
        self.roof_area_photo.setPixmap(QtGui.QPixmap(latest_image))

    def show_final_results(self):
        self.close()

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("Assessment of Civil Structures")
        # self.window_width, self.window_height = 1800, 600
        # self.setMinimumSize(self.window_width, self.window_height)
        self.setWindowState(Qt.WindowMaximized)
        self.main_layout = QVBoxLayout()
        

        self.video_path = QPushButton('Video Path')
        self.video_path.clicked.connect(self.getFileName)
        self.main_layout.addWidget(self.video_path)

        self.distance_modules = QHBoxLayout()
        self.distance_modules.addStretch(1)

        self.frontal_mode = QVBoxLayout()
        self.frontal_mode_photo = QtWidgets.QLabel()
        self.frontal_mode_photo.setPixmap(QtGui.QPixmap("./gui_images/frontalviewdronepov.jpg"))
        self.frontal_mode.addWidget(self.frontal_mode_photo)
        self.frontal_mode_button = QPushButton('Frontal Mode')
        self.frontal_mode.addWidget(self.frontal_mode_button)
        self.frontal_mode.setSpacing(0)
        self.distance_modules.addLayout(self.frontal_mode)


        self.inbetween_mode = QVBoxLayout()
        self.inbetween_mode_photo = QtWidgets.QLabel()
        self.inbetween_mode_photo.setPixmap(QtGui.QPixmap("./gui_images/inbetweenviewdronepov.jpg"))
        self.inbetween_mode.addWidget(self.inbetween_mode_photo)
        self.inbetween_mode_button = QPushButton('In-Between Mode')
        self.inbetween_mode.addWidget(self.inbetween_mode_button)
        self.inbetween_mode.setSpacing(0)
        self.distance_modules.addLayout(self.inbetween_mode)

        self.roof_mode = QVBoxLayout()
        self.roof_mode_photo = QtWidgets.QLabel()
        self.roof_mode_photo.setPixmap(QtGui.QPixmap("./gui_images/rooftopdronepov.jpg"))
        self.roof_mode.addWidget(self.roof_mode_photo)
        self.roof_mode_button = QPushButton('Roof Mode')
        self.roof_mode.setSpacing(0)
        self.roof_mode.addWidget(self.roof_mode_button)
        self.distance_modules.addLayout(self.roof_mode)

        self.distance_modules.addStretch(1)
        self.distance_modules.setSpacing(120)
        
        self.other_modules = QHBoxLayout()
        self.other_modules.addStretch(1)
        self.roofarea = QVBoxLayout()
        self.roofarea_photo = QtWidgets.QLabel()
        self.roofarea_photo.setPixmap(QtGui.QPixmap("./gui_images/roofareacalculation.png"))
        self.roofarea.addWidget(self.roofarea_photo)
        self.roof_area_button = QPushButton('Roof Area Calculation')
        self.roofarea.addWidget(self.roof_area_button)
        self.roof_area_button.clicked.connect(self.roof_area_calculation)
        self.roofarea.setSpacing(0)
        
        self.other_modules.addLayout(self.roofarea)

        self.roof_layout = QVBoxLayout()
        self.roof_layout_photo = QtWidgets.QLabel()
        self.roof_layout_photo.setPixmap(QtGui.QPixmap("./gui_images/rooflayoutestimation.png"))
        self.roof_layout.addWidget(self.roof_layout_photo)
        self.roof_layout_button = QPushButton('Roof Layout Estimation')
        self.roof_layout.setSpacing(0)
        self.roof_layout.addWidget(self.roof_layout_button)
        
        self.other_modules.addLayout(self.roof_layout)
        self.other_modules.addStretch(1)
        self.other_modules.setSpacing(80)

        self.main_layout.addLayout(self.distance_modules)
        self.main_layout.addLayout(self.other_modules)
        self.main_layout.setSpacing(100)
        self.main_layout.addStretch(1)

        widget = QWidget()
        widget.setLayout(self.main_layout)
        self.setCentralWidget(widget)
    
    def roof_area_calculation(self):
        self.close()


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
            font-size: 24px;
        }
    ''')
    
    myApp = MainWindow()
    myApp.show()
   
    try:
        sys.exit(app.exec_())
    except SystemExit:
        app1 = QApplication(sys.argv)
        myApp1 = DisplayResults()
        myApp1.show()
        sys.exit(app1.exec_())


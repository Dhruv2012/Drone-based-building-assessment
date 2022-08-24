from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import sys
import cv2
import os
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt,QObject
import glob
import time


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("Assessment of Civil Structures")
        self.window_width, self.window_height = 1800, 600
        self.setMinimumSize(self.window_width, self.window_height)

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
        roof_area_button = QPushButton('Roof Area Calculation')
        self.roofarea.addWidget(roof_area_button)
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
    
    def submitDialog(self):
        global mode
        option = self.options.index(self.combo.currentText())
        mode = self.options[option]
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
        print("")


# class DisplayResults(object):
#     def setup(self, MainWindow):
#         MainWindow.setObjectName("Assessment of Civil Structures")
#         MainWindow.resize(800, 600)
#         self.centralwidget = QtWidgets.QWidget(MainWindow)
#         self.centralwidget.setObjectName("centralwidget")
#         self.photo = QtWidgets.QLabel(self.centralwidget)
#         self.photo.setGeometry(QtCore.QRect(0, 0, 841, 511))
#         self.photo.setText("")
#         self.photo.setPixmap(QtGui.QPixmap("./test_folder/env0_easy_00002.png"))
#         self.photo.setScaledContents(True)
#         self.photo.setObjectName("photo")
#         self.intermediate_results = QtWidgets.QPushButton(self.centralwidget)
#         self.intermediate_results.setGeometry(QtCore.QRect(0, 510, 411, 41))
#         self.intermediate_results.setObjectName("Show Intermediate Results")
#         self.final_results = QtWidgets.QPushButton(self.centralwidget)
#         self.final_results.setGeometry(QtCore.QRect(410, 510, 391, 41))
#         self.final_results.setObjectName("Wait for Final Results")
#         MainWindow.setCentralWidget(self.centralwidget)
#         self.menubar = QtWidgets.QMenuBar(MainWindow)
#         self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
#         self.menubar.setObjectName("menubar")
#         MainWindow.setMenuBar(self.menubar)
#         self.statusbar = QtWidgets.QStatusBar(MainWindow)
#         self.statusbar.setObjectName("statusbar")
#         MainWindow.setStatusBar(self.statusbar)

#         self.retranslateUi(MainWindow)
#         QtCore.QMetaObject.connectSlotsByName(MainWindow)

#         self.intermediate_results.clicked.connect(self.show_intermediate_results)
#         self.final_results.clicked.connect(MainWindow.close)

#     def retranslateUi(self, MainWindow):
#         _translate = QtCore.QCoreApplication.translate
#         MainWindow.setWindowTitle(_translate("Assessment of Civil Structures", "Assessment of Civil Structures"))
#         self.final_results.setText(_translate("Assessment of Civil Structures", "Skip to Final Results"))
#         self.intermediate_results.setText(_translate("Assessment of Civil Structures", "Show Intermediate Results"))

#     def show_intermediate_results(self):
#         folder_path = './test_folder'
#         latest_image = max(glob.iglob(folder_path + '/*.png'), key=os.path.getctime)
#         print(latest_image)
#         if latest_image == './test_folder/env0_easy_00005.png':
#             _translate = QtCore.QCoreApplication.translate
#             self.intermediate_results.setText(_translate("Assessment of Civil Structures", "All Intermediate Results Displayed. Press Again to Quit."))
#             self.intermediate_results.clicked.connect(MainWindow.close)
#         self.photo.setPixmap(QtGui.QPixmap(latest_image))

#     def show_final_results(self):
#         self.close()




# if __name__ == "__main__":

#     app = QtWidgets.QApplication(sys.argv)
#     MainWindow = QtWidgets.QMainWindow()
#     ui = DisplayResults()
#     ui.setup(MainWindow)
#     MainWindow.show()
#     sys.exit(app.exec_())
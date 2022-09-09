from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import sys
import cv2
import os
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt,QObject
import glob
import docker
# client = docker.from_env()
class DisplayResults(QScrollArea):
    def __init__(self,mode):
        super(DisplayResults,self).__init__()
        
        self.setWindowTitle("Displaying Intermediate Results")
        # self.window_width, self.window_height = 1800, 600
        
        # self.setWindowState(Qt.WindowMaximized)
        self.window_height, self.window_width, _ = cv2.imread('./gui_images/'+mode+'.png').shape
        self.setMinimumSize(self.window_width+80, self.window_height+80)
        self.mode = mode
        self.main_layout = QVBoxLayout()
        self.main_layout.setAlignment(Qt.AlignTop)
        self.mode_photo = QtWidgets.QLabel()
        self.mode_photo.setPixmap(QtGui.QPixmap(os.path.join("./gui_images", self.mode+'.png')))
        self.mode_photo.setScaledContents(True)
        self.main_layout.addWidget(self.mode_photo)

        folder_path = self.mode
        self.intermediate_results_path = os.path.join(folder_path, 'intermediate_results')
        self.final_results_path = os.path.join(folder_path, 'final_results')
        os.makedirs(self.intermediate_results_path, exist_ok=True)
        os.makedirs(self.final_results_path, exist_ok=True)

        self.buttons = QHBoxLayout()

        self.intermediate_button = QPushButton('Show Intermediate Results')
        self.intermediate_button.clicked.connect(self.show_intermediate_results)
        self.buttons.addWidget(self.intermediate_button)
        
        self.final_button = QPushButton('Show Final Results')
        self.final_button.clicked.connect(self.show_final_results)
        self.buttons.addWidget(self.final_button)

        self.main_layout.addLayout(self.buttons)
        self.main_layout.setSpacing(0)

        widget = QWidget()
        widget.setLayout(self.main_layout)
        self.setWidget(widget)
        self.setWidgetResizable(True)
    
    def retranslateUi(self, DisplayResults):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("Displaying Intermediate Results", "Displaying Intermediate Results"))
        self.final_button.setText(_translate("Displaying Intermediate Results", "Skip to Final Results"))
        self.intermediate_button.setText(_translate("Displaying Intermediate Results", "Show Intermediate Results"))

    def show_intermediate_results(self):


        latest_image = max(glob.iglob(self.intermediate_results_path + '/*'), key=os.path.getctime)
        if latest_image == self.intermediate_results_path+'/done.txt':
            self.mode_photo.setPixmap(QtGui.QPixmap(os.path.join("./gui_images", self.mode+'.png')))
            _translate = QtCore.QCoreApplication.translate
            
            self.intermediate_button.setText(_translate("Displaying Intermediate Results", "All Intermediate Results Displayed. Press Again to Quit."))
            self.intermediate_button.clicked.connect(self.close)
        self.mode_photo.setPixmap(QtGui.QPixmap(latest_image))

    def show_final_results(self):
        self.close()

class MainWindow(QScrollArea):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.myApp = None
        self.setWindowTitle("Assessment of Civil Structures")
        self.setWindowState(Qt.WindowMaximized)
        self.main_layout = QVBoxLayout()
        self.main_layout.setAlignment(Qt.AlignTop)
        

        self.video_path = QPushButton('Video Path')
        self.video_path.clicked.connect(self.GetVideoFile)
        self.main_layout.addWidget(self.video_path)
        self.log_path = QPushButton('Log Path')
        self.log_path.clicked.connect(self.GetLogFile)
        self.main_layout.addWidget(self.log_path)

        self.distance_modules = QHBoxLayout()
        self.distance_modules.addStretch(1)

        self.frontal_mode = QVBoxLayout()
        self.frontal_mode_photo = QtWidgets.QLabel()
        self.frontal_mode_photo.setPixmap(QtGui.QPixmap("./gui_images/frontalviewdronepov.jpg"))
        self.frontal_mode.addWidget(self.frontal_mode_photo)
        self.frontal_mode_button = QPushButton('Frontal Mode')
        self.frontal_mode_button.clicked.connect(self.frontal_mode_results)
        self.frontal_mode.addWidget(self.frontal_mode_button)
        self.frontal_mode.setSpacing(0)
        self.distance_modules.addLayout(self.frontal_mode)


        self.inbetween_mode = QVBoxLayout()
        self.inbetween_mode_photo = QtWidgets.QLabel()
        self.inbetween_mode_photo.setPixmap(QtGui.QPixmap("./gui_images/inbetweenviewdronepov.jpg"))
        self.inbetween_mode.addWidget(self.inbetween_mode_photo)
        self.inbetween_mode_button = QPushButton('In-Between Mode')
        self.inbetween_mode_button.clicked.connect(self.inbetween_mode_results)
        self.inbetween_mode.addWidget(self.inbetween_mode_button)
        self.inbetween_mode.setSpacing(0)
        self.distance_modules.addLayout(self.inbetween_mode)

        self.roof_mode = QVBoxLayout()
        self.roof_mode_photo = QtWidgets.QLabel()
        self.roof_mode_photo.setPixmap(QtGui.QPixmap("./gui_images/rooftopdronepov.jpg"))
        self.roof_mode.addWidget(self.roof_mode_photo)
        self.roof_mode_button = QPushButton('Roof Mode')
        self.roof_mode_button.clicked.connect(self.roof_mode_results)
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
        self.roof_layout_button.clicked.connect(self.roof_layout_estimation)
        self.roof_layout.setSpacing(0)
        self.roof_layout.addWidget(self.roof_layout_button)
        
        self.other_modules.addLayout(self.roof_layout)
        self.other_modules.addStretch(1)
        self.other_modules.setSpacing(50)

        self.contributors_list = QLabel()
        self.contributors_list.setText("<html><ul>Contributors:<li>Kushagra Srivastava</li><li>Dhruv Patel</li><li>Aditya Kumar Jha</li><li>Mohhit Kumar Jha</li><li>Ekansh Gupta</li><li>Jaskirat Singh</li><li>Ravi Kiran Sarvadevabhatla</li><li>Pradeep Kumar Ramancharla</li><li>Harikumar Kandath</li><li>K. Madhava Krishna</li></ul></html>")

        self.main_layout.addLayout(self.distance_modules)
        self.main_layout.addLayout(self.other_modules)
        self.main_layout.addWidget(self.contributors_list)
        self.main_layout.setSpacing(100)
        self.main_layout.addStretch(1)

        widget = QWidget()
        widget.setLayout(self.main_layout)
        self.setWidget(widget)
        self.setWidgetResizable(True)
    
    def roof_area_calculation(self,checked):
        self.close()
        # container = client.containers.get('4cd0a2253a46')
        # container.exec_run('python3 test.py --datadir ../../../images/roofimages --resultdir ../../../RoofResults', workdir='/root/RoofSegmentation/LEDNet/test/')
        if self.myApp is None:
            self.myApp = DisplayResults(mode="RoofAreaCalculation")
        self.myApp.show()


    
    def roof_layout_estimation(self,checked):
        self.close()
        if self.myApp is None:
            self.myApp = DisplayResults(mode="RoofLayoutEstimation")
        self.myApp.show()
    
    def frontal_mode_results(self,checked):
        self.close()
        if self.myApp is None:
            self.myApp = DisplayResults(mode="DistanceModuleFrontalMode")
        self.myApp.show()

    def inbetween_mode_results(self,checked):
        self.close()
        if self.myApp is None:
            self.myApp = DisplayResults(mode="DistanceModuleInBetweenMode")
        self.myApp.show()
    
    def roof_mode_results(self,checked):
        self.close()
        if self.myApp is None:
            self.myApp = DisplayResults(mode="DistanceModuleRoofMode")
        self.myApp.show()
    
    def GetVideoFile(self):
        global videopath
        response = QFileDialog.getOpenFileName(self, str("Select a video file"),
                                       os.getcwd(),
                                       str("Videos (*.mp4 *.mov)"))
        videopath = response
    
    def GetLogFile(self):
        global logpath
        response = QFileDialog.getOpenFileName(self, str("Select a log file"),
                                       os.getcwd(),
                                       str("Logs (*.csv)"))
        logpath = response
if __name__ == '__main__':

    app = QApplication(sys.argv)
    app.setStyleSheet('''QWidget {font-size: 24px; }''')
    
    myApp = MainWindow()
    myApp.show()
    app.exec_()

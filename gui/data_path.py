import datetime
import time
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import sys
import cv2
import os
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt,QObject, QRect, QThread, QMutex
import glob
import docker
import shutil
import subprocess
# import threading
# client = docker.from_env()
line = ""
mutex = QMutex()
class DisplayFinalResults(QWidget):

  def __init__(self, mode):
    super(DisplayFinalResults, self).__init__()
    layout = QVBoxLayout()
    self.label = QLabel()
    self.setMinimumSize(500,500)
    layout.addWidget(self.label)
    self.mode = mode
    self.content = QTextEdit()
    layout.addWidget(self.content)
    self.setWindowTitle("Displaying Final Results for " + self.mode)
    self.setLayout(layout)
    self.load_text()
  
  def load_text(self):

    filenames = glob.glob(os.path.join(self.mode, 'final_results/*.txt'))
    f = open(filenames[0], 'r')
    with f:
        data = f.read()
        self.content.setText(data)

class Worker(QObject):
    
    finished = pyqtSignal()
    progress = pyqtSignal()
    def __init__(self, mode):

        super(Worker, self).__init__()
        self.mode = mode
        self.log_file = os.path.join(self.mode, 'log.txt')
        # self.log_file = 'log.txt'

    def showlogs(self):

        global line
        with open(self.log_file, 'r') as f:
            line=f.read()
            self.progress.emit()
            if os.path.exists(os.path.join(self.mode, 'final_results/final_results.txt')):
                f.close()
        self.finished.emit()
        
class DisplayResults(QScrollArea):
    def __init__(self,mode):
        super(DisplayResults,self).__init__()
        
        self.setWindowTitle("Displaying Intermediate Results")
        # self.window_width, self.window_height = 1800, 600
        self.DisplayFinalResultsApp = None
        # self.setWindowState(Qt.WindowMaximized)
        self.window_height, self.window_width, _ = cv2.imread('./gui_images/'+mode+'.png').shape
        self.setMinimumSize(self.window_width+300, self.window_height+80)
        self.mode = mode
        
        self.layout_final = QHBoxLayout()
        self.layout_final.setSpacing(0)
        self.main_layout = QVBoxLayout()
        self.main_layout.setAlignment(Qt.AlignTop)
        self.mode_photo = QtWidgets.QLabel()
        self.mode_photo.setPixmap(QtGui.QPixmap(os.path.join("./gui_images", self.mode+'.png')))
        self.mode_photo.setScaledContents(True)
        # self.mode_photo.setScaledContents(True)
        self.mode_photo.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.mode_photo)

        folder_path = self.mode
        self.intermediate_results_path = os.path.join(folder_path, 'intermediate_results')
        self.final_results_path = os.path.join(folder_path, 'final_results')
        
        if os.path.exists(self.intermediate_results_path):
            files = os.listdir(self.intermediate_results_path)
            for f in files:
                os.remove(os.path.join(self.intermediate_results_path,f))
        else:
            os.makedirs(self.intermediate_results_path)
        
        if os.path.exists(self.final_results_path):
            files = os.listdir(self.final_results_path)
            for f in files:
                os.remove(os.path.join(self.final_results_path,f))
        else:
            os.makedirs(self.final_results_path)
        
        with open(os.path.join(self.mode,'log.txt') , 'w') as f:
            f.close()

        shutil.copy(os.path.join("./gui_images", self.mode+'.png'), self.intermediate_results_path)

        self.buttons = QHBoxLayout()
        self.final_button = QPushButton('Show Final Results')
        self.final_button.clicked.connect(self.ShowFinalResults)
        self.final_button.setDisabled(True)
        self.buttons.addWidget(self.final_button)

        self.main_layout.addLayout(self.buttons)
        self.main_layout.setSpacing(0)
        self.layout_final.addLayout(self.main_layout)


        self.progress = QLabel()
        self.layout_final.addWidget(self.progress)
        self.content = QTextEdit()
        self.scroll_bar = self.content.verticalScrollBar()

        self.scroll_bar.setValue(self.scroll_bar.maximum())
        self.layout_final.addWidget(self.content)
        self.layout_final.setSpacing(0)

        widget = QWidget()
        widget.setLayout(self.layout_final)
        self.setWidget(widget)
        self.setWidgetResizable(True)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.SetLogThread)
        self.timer.start(1000)
        
        self.timer_intermediate = QtCore.QTimer(self)
        self.timer_intermediate.timeout.connect(self.ShowIntermediateResults)
        self.timer_intermediate.start(1000)

        self.p=subprocess.Popen("./"+self.mode+'.sh', shell=True)
    

    def ShowIntermediateResults(self):

        latest_image = max(glob.iglob(self.intermediate_results_path + '/*'), key=os.path.getctime)
        self.mode_photo.setPixmap(QtGui.QPixmap(latest_image))
        self.mode_photo.setScaledContents(True)


    def ShowFinalResults(self):
        # For testing purposes
        # with open(self.final_results_path+'/final_results1.txt', 'w') as f:
        #     f.write('done\n Final Results shown')
        latest_file = glob.glob(self.final_results_path + '/*')
        if len(latest_file) == 0:
            pass
        elif 'final_results' in latest_file[0].split('/')[-1]:
            self.p.kill()
            self.final_button.setDisabled(False)
            self.final_button.setText('Click Here View Final Results')
            self.final_button.clicked.connect(self.DisplayFinalResultsWindow)

    def SetLogThread(self):

        self.thread = QThread()
        self.worker = Worker(self.mode)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.showlogs)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()
        self.thread.finished.connect(self.ShowFinalResults)
        self.worker.progress.connect(self.DisplayLogs)
    
    def DisplayLogs(self):
        self.content.setText(line)
        self.scroll_bar.setValue(self.scroll_bar.maximum())
    
    def DisplayFinalResultsWindow(self):

        self.DisplayFinalResultsApp = DisplayFinalResults(mode=self.mode)
        self.DisplayFinalResultsApp.show()
        # self.close()

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

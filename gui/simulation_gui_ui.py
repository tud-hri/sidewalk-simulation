"""
Copyright 2024, Olger Siebinga (o.siebinga@tudelft.nl)

This file is part of sidewalk-simulation.

sidewalk-simulation is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

sidewalk-simulation is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with sidewalk-simulation.  If not, see <https://www.gnu.org/licenses/>.
"""
# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'simulation_gui.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_SimpleMerging(object):
    def setupUi(self, SimpleMerging):
        SimpleMerging.setObjectName("SimpleMerging")
        SimpleMerging.resize(650, 650)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(SimpleMerging.sizePolicy().hasHeightForWidth())
        SimpleMerging.setSizePolicy(sizePolicy)
        SimpleMerging.setMinimumSize(QtCore.QSize(650, 650))
        self.centralwidget = QtWidgets.QWidget(SimpleMerging)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.world_view = WorldView(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.world_view.sizePolicy().hasHeightForWidth())
        self.world_view.setSizePolicy(sizePolicy)
        self.world_view.setMinimumSize(QtCore.QSize(600, 420))
        self.world_view.setObjectName("world_view")
        self.verticalLayout.addWidget(self.world_view)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.previous_button = QtWidgets.QPushButton(self.centralwidget)
        self.previous_button.setObjectName("previous_button")
        self.horizontalLayout_3.addWidget(self.previous_button)
        self.play_button = QtWidgets.QPushButton(self.centralwidget)
        self.play_button.setObjectName("play_button")
        self.horizontalLayout_3.addWidget(self.play_button)
        self.next_button = QtWidgets.QPushButton(self.centralwidget)
        self.next_button.setObjectName("next_button")
        self.horizontalLayout_3.addWidget(self.next_button)
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setObjectName("groupBox")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.groupBox)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label_7 = QtWidgets.QLabel(self.groupBox)
        self.label_7.setObjectName("label_7")
        self.horizontalLayout_5.addWidget(self.label_7)
        self.showAgent0BeliefCheckBox = QtWidgets.QCheckBox(self.groupBox)
        self.showAgent0BeliefCheckBox.setObjectName("showAgent0BeliefCheckBox")
        self.horizontalLayout_5.addWidget(self.showAgent0BeliefCheckBox)
        self.showAgent0PlanCheckBox = QtWidgets.QCheckBox(self.groupBox)
        self.showAgent0PlanCheckBox.setObjectName("showAgent0PlanCheckBox")
        self.horizontalLayout_5.addWidget(self.showAgent0PlanCheckBox)
        self.label_6 = QtWidgets.QLabel(self.groupBox)
        self.label_6.setObjectName("label_6")
        self.horizontalLayout_5.addWidget(self.label_6)
        self.showAgent1BeliefCheckBox = QtWidgets.QCheckBox(self.groupBox)
        self.showAgent1BeliefCheckBox.setObjectName("showAgent1BeliefCheckBox")
        self.horizontalLayout_5.addWidget(self.showAgent1BeliefCheckBox)
        self.showAgent1PlanCheckBox = QtWidgets.QCheckBox(self.groupBox)
        self.showAgent1PlanCheckBox.setObjectName("showAgent1PlanCheckBox")
        self.horizontalLayout_5.addWidget(self.showAgent1PlanCheckBox)
        self.horizontalLayout_3.addWidget(self.groupBox)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.timeSlider = QtWidgets.QSlider(self.centralwidget)
        self.timeSlider.setMaximum(999)
        self.timeSlider.setOrientation(QtCore.Qt.Horizontal)
        self.timeSlider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.timeSlider.setObjectName("timeSlider")
        self.verticalLayout.addWidget(self.timeSlider)
        self.horizontalLayout_2.addLayout(self.verticalLayout)
        SimpleMerging.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(SimpleMerging)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 650, 21))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        SimpleMerging.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(SimpleMerging)
        self.statusbar.setObjectName("statusbar")
        SimpleMerging.setStatusBar(self.statusbar)
        self.actionEnable_recording = QtWidgets.QAction(SimpleMerging)
        self.actionEnable_recording.setCheckable(True)
        self.actionEnable_recording.setObjectName("actionEnable_recording")
        self.menuFile.addAction(self.actionEnable_recording)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(SimpleMerging)
        QtCore.QMetaObject.connectSlotsByName(SimpleMerging)

    def retranslateUi(self, SimpleMerging):
        _translate = QtCore.QCoreApplication.translate
        SimpleMerging.setWindowTitle(_translate("SimpleMerging", "Simple Merging"))
        self.previous_button.setText(_translate("SimpleMerging", "<"))
        self.play_button.setText(_translate("SimpleMerging", "Start"))
        self.next_button.setText(_translate("SimpleMerging", ">"))
        self.groupBox.setTitle(_translate("SimpleMerging", "Plan and Belief Visualisation in World"))
        self.label_7.setText(_translate("SimpleMerging", "Agent 0:"))
        self.showAgent0BeliefCheckBox.setText(_translate("SimpleMerging", "Belief"))
        self.showAgent0PlanCheckBox.setText(_translate("SimpleMerging", "Plan"))
        self.label_6.setText(_translate("SimpleMerging", "Agent 1: "))
        self.showAgent1BeliefCheckBox.setText(_translate("SimpleMerging", "Belief"))
        self.showAgent1PlanCheckBox.setText(_translate("SimpleMerging", "Plan"))
        self.menuFile.setTitle(_translate("SimpleMerging", "File"))
        self.actionEnable_recording.setText(_translate("SimpleMerging", "Enable recording"))
from gui.worldview import WorldView
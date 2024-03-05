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
import datetime
import os

import numpy as np
import pyqtgraph
from PyQt5 import QtWidgets, QtCore, QtGui

from .simulation_gui_ui import Ui_SimpleMerging


class SimulationGui(QtWidgets.QMainWindow):
    def __init__(self, track, in_replay_mode=False, number_of_belief_points=0, enable_print_style_plots=False,
                 parent=None):
        super().__init__(parent)

        if enable_print_style_plots:
            pyqtgraph.setConfigOption('background', 'w')
            pyqtgraph.setConfigOption('foreground', 'k')

        self.ui = Ui_SimpleMerging()
        self.ui.setupUi(self)

        self.ui.world_view.initialize(track, self, number_of_belief_points)

        self.track = track
        self.in_replay_mode = in_replay_mode

        self.ui.play_button.clicked.connect(self.toggle_play)
        self.ui.previous_button.clicked.connect(self.previous_frame)
        self.ui.next_button.clicked.connect(self.next_frame)
        self.ui.next_button.setEnabled(False)
        self.ui.previous_button.setEnabled(False)
        self.ui.timeSlider.setEnabled(in_replay_mode)
        self.ui.timeSlider.sliderReleased.connect(self._set_time)

        self.ui.showAgent0PlanCheckBox.stateChanged.connect(self._toggle_world_view_plots)
        self.ui.showAgent0BeliefCheckBox.stateChanged.connect(self._toggle_world_view_plots)
        self.ui.showAgent1PlanCheckBox.stateChanged.connect(self._toggle_world_view_plots)
        self.ui.showAgent1BeliefCheckBox.stateChanged.connect(self._toggle_world_view_plots)

        self.is_expanded = True

        self._time_indicator_lines = []
        self._distance_indicator_lines = []

        self.video_writer = None
        self.is_recording = False
        self.path_to_video_file = ''

        self.ui.actionEnable_recording.triggered.connect(self._enable_recording)

        self.show()

    def _expand_window(self):
        if self.is_expanded:
            self.ui.tabWidget.setVisible(False)
            self.ui.expandPushButton.setText('>')
            self.resize(650, 650)
        else:
            self.ui.tabWidget.setVisible(True)
            self.ui.expandPushButton.setText('<')
            self.resize(1300, 650)

        self.is_expanded = not self.is_expanded

    def register_sim_master(self, sim_master):
        self.sim_master = sim_master

    def toggle_play(self):
        if self.sim_master and not self.sim_master.main_timer.isActive():
            self.sim_master.start()
            if self.in_replay_mode:
                self.ui.play_button.setText('Pause')
                self.ui.next_button.setEnabled(False)
                self.ui.previous_button.setEnabled(False)
            else:
                self.ui.play_button.setEnabled(False)
        elif self.sim_master:
            self.sim_master.pause()
            self.ui.play_button.setText('Play')
            if self.in_replay_mode:
                self.ui.next_button.setEnabled(True)
                self.ui.previous_button.setEnabled(True)

    def reset(self):
        self.ui.play_button.setText('Play')
        self.ui.play_button.setEnabled(True)
        if self.in_replay_mode:
            self.ui.next_button.setEnabled(True)
            self.ui.previous_button.setEnabled(True)

    def next_frame(self):
        if self.in_replay_mode:
            self.sim_master.do_time_step()

    def previous_frame(self):
        if self.in_replay_mode:
            self.sim_master.do_time_step(reverse=True)

    def add_controllable_dot(self, controllable_object, color=QtCore.Qt.red):
        self.ui.world_view.add_controllable_dot(controllable_object, color)

    def update_all_graphics(self):
        self.ui.world_view.update_all_graphics_positions()

    def _toggle_world_view_plots(self):
        self.ui.world_view.set_plan_visible(0, self.ui.showAgent0PlanCheckBox.isChecked())
        self.ui.world_view.set_plan_visible(1, self.ui.showAgent1PlanCheckBox.isChecked())
        self.ui.world_view.set_belief_visible(0, self.ui.showAgent0BeliefCheckBox.isChecked())
        self.ui.world_view.set_belief_visible(1, self.ui.showAgent1BeliefCheckBox.isChecked())

    def update_plots(self, agent, belief, belief_time_stamps, position_plan):
        if (agent == 0 and (self.ui.showAgent0PlanCheckBox.isChecked() or self.ui.showAgent0BeliefCheckBox.isChecked())) or (
                agent == 1 and (self.ui.showAgent1PlanCheckBox.isChecked() or self.ui.showAgent1BeliefCheckBox.isChecked())):
            belief_times_from_now = np.array(belief_time_stamps)
            position_indices = ((belief_times_from_now / (self.sim_master.dt / 1000.)) - 1).astype(int)

            position_plan_corresponding_to_belief = position_plan[position_indices]
            self.ui.world_view.update_plan_and_belief_graphics(agent, position_plan_corresponding_to_belief, belief)

    def update_time_label(self, time):
        self.ui.statusbar.showMessage('time: %0.2f s' % time)
        if self.in_replay_mode:
            time_promille = int(self.sim_master.time_index * 1000 / self.sim_master.maxtime_index)
            self.ui.timeSlider.setValue(time_promille)

    def show_overlay(self, message=None):
        if message:
            self.ui.world_view.set_overlay_message(message)
            self.ui.world_view.draw_overlay(True)
        else:
            self.ui.world_view.draw_overlay(False)

    def _set_time(self):
        time_promille = self.ui.timeSlider.value()
        self.sim_master.set_time(time_promille=time_promille)

    @staticmethod
    def _add_padding_to_plot_widget(plot_widget, padding=0.1):
        """
        zooms out the view of a plot widget to show 'padding' around the contents of a PlotWidget
        :param plot_widget: The widget to add padding to
        :param padding: the percentage of padding expressed between 0.0 and 1.0
        :return:
        """

        width = plot_widget.sceneRect().width() * (1. + padding)
        height = plot_widget.sceneRect().height() * (1. + padding)
        center = plot_widget.sceneRect().center()
        zoom_rect = QtCore.QRectF(center.x() - width / 2., center.y() - height / 2., width, height)

        plot_widget.fitInView(zoom_rect)

    def _enable_recording(self):
        if not self.is_recording:
            self.initialize_recording()
        else:
            self.stop_recording()

    def initialize_recording(self):
        file_name = datetime.datetime.now().strftime('%Y%m%d-%Hh%Mm%Ss.avi')

        self.path_to_video_file = os.path.join('data', 'videos', file_name)
        fps = 1 / (self.sim_master.dt / 1000.)

        frame_size = self._get_image_of_current_gui().size()
        self.video_writer = cv2.VideoWriter(self.path_to_video_file, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'), fps, (frame_size.width(), frame_size.height()))
        self.is_recording = True
        self.sim_master.enable_recording(True)

    def stop_recording(self):
        self.video_writer.release()
        QtWidgets.QMessageBox.information(self, 'Video Saved', 'A video capture of the visualisation was saved to ' + self.path_to_video_file)
        self.is_recording = False

    def record_frame(self):
        if self.is_recording:
            image = self._get_image_of_current_gui()
            frame_size = image.size()
            bits = image.bits()

            bits.setsize(frame_size.height() * frame_size.width() * 4)
            image_array = np.frombuffer(bits, np.uint8).reshape((frame_size.height(), frame_size.width(), 4))
            color_convert_image = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
            self.video_writer.write(color_convert_image)

    def _save_screen_shot(self):
        time_stamp = datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')
        file_name = os.path.join('data', 'images', time_stamp + '.png')

        image = self._get_image_of_current_gui()
        image.save(file_name)

    def _get_image_of_current_gui(self):
        image = QtGui.QImage(self.size(), QtGui.QImage.Format_ARGB32_Premultiplied)
        region = QtGui.QRegion(self.rect())

        painter = QtGui.QPainter(image)
        self.render(painter, QtCore.QPoint(), region)
        painter.end()

        return image

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
import os
import random

import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore


class WorldView(QtWidgets.QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_gui = None
        self.sidewalk_graphics = None
        self.track = None

        self.scene = QtWidgets.QGraphicsScene()
        self.setRenderHint(QtGui.QPainter.HighQualityAntialiasing)
        self.setScene(self.scene)
        self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(14, 150, 22)))

        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)

        # Scaled size zoomRect
        self.max_zoom_size = 0.0
        self.min_zoom_size = 0.0
        self.zoom_level = 0.0
        self.zoom_center = None

        # intialize overlay
        self._overlay_message = 'Not started'
        self._draw_overlay = True

        self.controllable_objects = []
        self.graphics_objects = []

        # initialize plan and believe point lists
        self.plan_graphics_objects = {0: [],
                                      1: []}
        self.belief_graphics_objects = {0: [],
                                        1: []}

    def initialize(self, track, main_gui, number_of_belief_points):
        self.main_gui = main_gui
        self.sidewalk_graphics = QtWidgets.QGraphicsItemGroup()
        self.track = track

        sidewalk_path = QtWidgets.QGraphicsPathItem()
        sidewalk_painter = QtGui.QPainterPath()
        pen = QtGui.QPen()
        pen.setWidthF(track.track_width)
        pen.setColor(QtGui.QColor(150, 150, 150))
        sidewalk_path.setPen(pen)

        sidewalk_painter.moveTo(track.get_way_points()[0][0], -track.get_way_points()[0][1])

        for way_point in track.get_way_points():
            sidewalk_painter.lineTo(way_point[0], -way_point[1])

        sidewalk_path.setPath(sidewalk_painter)

        self.sidewalk_graphics.addToGroup(sidewalk_path)

        self.sidewalk_graphics.setZValue(1.0)
        self.scene.addItem(self.sidewalk_graphics)
        padding_rect_size = self.sidewalk_graphics.sceneBoundingRect().size() * 4.0
        padding_rect_top_left_x = self.sidewalk_graphics.sceneBoundingRect().center().x() - padding_rect_size.width() / 2
        padding_rect_top_left_y = -self.sidewalk_graphics.sceneBoundingRect().center().y() - padding_rect_size.height() / 2
        scroll_padding_rect = QtWidgets.QGraphicsRectItem(padding_rect_top_left_x, padding_rect_top_left_y, padding_rect_size.width(),
                                                          padding_rect_size.height())
        scroll_padding_rect.setPen(QtGui.QPen(QtCore.Qt.NoPen))
        self.scene.addItem(scroll_padding_rect)

        # initialize belief and plan graphics
        colors = {0: QtGui.QColor(255, 127, 14),
                  1: QtGui.QColor(31, 119, 180)}

        for agent_index in [0, 1]:
            for belief_index in range(number_of_belief_points):
                plan_pen = QtGui.QPen()
                plan_pen.setColor(colors[agent_index])
                plan_pen.setWidthF(0.051)
                plan_graphics = QtWidgets.QGraphicsEllipseItem(-0.05, -0.05, .1, .1)
                plan_graphics.setPen(plan_pen)
                plan_graphics.setBrush(QtGui.QBrush(colors[agent_index]))

                plan_graphics.setVisible(False)
                plan_graphics.setZValue(3.0)
                self.plan_graphics_objects[agent_index].append(plan_graphics)

                line_height = .2
                belief_graphics = QtWidgets.QGraphicsRectItem(-self.track.track_width / 2, -line_height / 2, self.track.track_width, line_height)
                belief_graphics.setVisible(False)

                belief_brush = QtGui.QBrush(QtCore.Qt.magenta)
                belief_graphics.setZValue(2.5)
                belief_graphics.setBrush(belief_brush)
                belief_graphics.setPen(QtGui.QPen(QtCore.Qt.NoPen))
                self.belief_graphics_objects[agent_index].append(belief_graphics)

                self.scene.addItem(plan_graphics)
                self.scene.addItem(belief_graphics)

        # Scaled size zoomRect
        self.max_zoom_size = self.sidewalk_graphics.sceneBoundingRect().size() * 1
        self.min_zoom_size = self.sidewalk_graphics.sceneBoundingRect().size() * 0.01
        self.zoom_level = 0.0
        self.zoom_center = self.sidewalk_graphics.sceneBoundingRect().center()
        self.update_zoom()

    def add_controllable_dot(self, controllable_object, color=QtCore.Qt.red):
        radius = .25

        polygon = QtGui.QPolygonF([QtCore.QPointF(-radius, -radius),
                                   QtCore.QPointF(radius, 0.),
                                   QtCore.QPointF(-radius, radius)])
        graphics = QtWidgets.QGraphicsPolygonItem(polygon)
        # graphics = QtWidgets.QGraphicsEllipseItem(-radius, -radius, 2 * radius, 2 * radius)

        graphics.setPen(QtGui.QPen(QtCore.Qt.NoPen))
        graphics.setBrush(QtGui.QBrush(color))

        graphics.setZValue(2.0)
        self.scene.addItem(graphics)
        self.controllable_objects.append(controllable_object)
        self.graphics_objects.append(graphics)
        self.update_all_graphics_positions()

    def set_plan_visible(self, agent, boolean):
        for plan_graphics in self.plan_graphics_objects[agent]:
            plan_graphics.setVisible(boolean)

    def set_belief_visible(self, agent, boolean):
        for belief_graphics in self.belief_graphics_objects[agent]:
            belief_graphics.setVisible(boolean)

    def update_plan_and_belief_graphics(self, agent, position_plan, belief):
        for plan_graphics, plan_point in zip(self.plan_graphics_objects[agent], position_plan):
            plan_graphics.setPos(plan_point[0], -plan_point[1])

        for belief_graphics, belief_point in zip(self.belief_graphics_objects[agent], belief):
            belief_graphics.setPos(0., -belief_point.y)

            gradient_positions, gradient_values = (
                belief_point.get_normalized_values(slices=21, track_width=self.track.track_width))
            gradient = QtGui.QLinearGradient(-self.track.track_width / 2, 0., self.track.track_width / 2, 0.)

            for position, value in zip(gradient_positions, gradient_values):
                color = (int(value * 255), 127 - int(value * 45), 14, 127)
                color = QtGui.QColor(*color)
                gradient.setColorAt(position, color)

            brush = QtGui.QBrush(gradient)
            belief_graphics.setBrush(brush)

    def update_all_graphics_positions(self):
        for controllable_object, graphics_object in zip(self.controllable_objects, self.graphics_objects):
            graphics_object.setPos(controllable_object.position[0], -controllable_object.position[1])
            graphics_object.setRotation(-np.degrees(controllable_object.heading))

    def update_zoom(self):
        # Compute scale factors (in x- and y-direction)
        zoom = (1.0 - self.zoom_level) ** 2
        scale1 = zoom + (self.min_zoom_size.width() / self.max_zoom_size.width()) * (1.0 - zoom)
        scale2 = zoom + (self.min_zoom_size.height() / self.max_zoom_size.height()) * (1.0 - zoom)

        # Scaled size zoomRect
        scaled_w = self.max_zoom_size.width() * scale1
        scaled_h = self.max_zoom_size.height() * scale2

        # Set zoomRect
        view_zoom_rect = QtCore.QRectF(self.zoom_center.x() - scaled_w / 2, self.zoom_center.y() - scaled_h / 2, scaled_w, scaled_h)

        # Set view (including padding)
        self.fitInView(view_zoom_rect, QtCore.Qt.KeepAspectRatio)

    def set_overlay_message(self, message):
        self._overlay_message = message

    def draw_overlay(self, bool):
        self._draw_overlay = bool
        self.scene.update()

    def drawForeground(self, painter, rect):
        if self._draw_overlay:
            painter.setBrush(QtGui.QBrush(QtGui.QColor(100, 100, 100)))
            painter.setPen(QtGui.QPen())
            painter.setOpacity(0.3)

            # create rectangle with 20% margin around the edges for smooth panning
            corner = self.mapToScene(QtCore.QPoint(round(-0.2 * self.width()), round(-0.2 * self.height())))

            painter.drawRect(round(corner.x()), round(corner.y()), round(1.4 * self.width()), round(1.4 * self.height()))

            painter.setOpacity(1.0)
            font = QtGui.QFont()
            font.setPointSize(1)
            font.setLetterSpacing(QtGui.QFont.PercentageSpacing, 130.)

            painter_path = QtGui.QPainterPath()
            painter.setBrush(QtCore.Qt.white)
            painter.setPen(QtCore.Qt.NoPen)
            painter.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.TextAntialiasing)

            text_width = QtGui.QFontMetrics(font).horizontalAdvance(self._overlay_message)
            painter_path.addText(rect.center().x() - text_width / 2, rect.center().y(), font, self._overlay_message)

            painter.drawPath(painter_path)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.update_zoom()

    def wheelEvent(self, event):
        direction = np.sign(event.angleDelta().y())
        self.zoom_level = max(min(self.zoom_level + direction * 0.1, 1.0), 0.0)
        self.update_zoom()

    def enterEvent(self, e):
        self.viewport().setCursor(QtCore.Qt.ArrowCursor)
        super().enterEvent(e)

    def mouseMoveEvent(self, e):
        if e.buttons() == QtCore.Qt.LeftButton:  # Drag scene
            self.zoom_center = self.mapToScene(self.rect().center())
        super().mouseMoveEvent(e)

    def mousePressEvent(self, e):
        if e.buttons() == QtCore.Qt.MiddleButton:  # Drag scene
            self.main_gui.statusBar().showMessage('position of mouse: %0.1f, %0.1f  -  position of point mass: %0.1f, %0.1f  ' % (
                self.mapToScene(e.pos()).x(), -self.mapToScene(e.pos()).y(), self.controllable_objects[0].position[0],
                -self.controllable_objects[0].position[1]))
            print('b: %.2f, o: %.2f' % (self.controllable_objects[0].velocity, self.controllable_objects[1].velocity))
        super().mousePressEvent(e)

    def mouseReleaseEvent(self, e):
        super().mouseReleaseEvent(e)
        self.viewport().setCursor(QtCore.Qt.ArrowCursor)
        self.update_zoom()

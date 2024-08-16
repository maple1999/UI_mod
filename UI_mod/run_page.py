import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
import yaml
# from PyQt5.QtWidgets import QMessageBox, QFileDialog, QLineEdit
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import cv2
from matplotlib import pyplot as plt
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import pose as mp_pose
from inference_and_visualization_camera import main
import threading
from model import PoseRAC, Action_trigger
from screeninfo import get_monitors
import mediapipe as mp
import subprocess
from inference_and_visualization import PoseClassificationVisualizer, get_landmarks, normalize_landmarks
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

class VideoPlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Player with Action Counting")
        self.setGeometry(100, 100, 800, 600)

        # 创建一个 QLabel 显示视频帧
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)

        # 创建一个按钮用于上传视频
        self.upload_button = QPushButton("Upload Video", self)
        self.upload_button.clicked.connect(self.load_video)

        # 布局
        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.upload_button)

        # 创建一个中央小部件并设置布局
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # 定时器用于更新视频帧
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # 视频捕获对象
        self.cap = None
        self.tensors = None
        self.frame_count = 0
        self.init_pose = 'pose_holder'
        self.curr_pose = 'holder'
        self.pose_count = 0
        self.classify_prob = 0.5
        self.momentum = 0.4

        # 加载动作标签和模型
        self.load_labels_and_model()

    def load_labels_and_model(self):
        csv_label_path = '..\\all_action_realtime.csv'
        label_pd = pd.read_csv(csv_label_path)
        self.index2action = {}
        length_label = len(label_pd.index)
        for label_i in range(length_label):
            one_data = label_pd.iloc[label_i]
            action = one_data['action']
            label = one_data['label']
            self.index2action[label] = action

        self.num_classes = len(self.index2action)
        self.real_index = 0  # 假设你要检测的动作索引
        self.action_type = self.index2action[self.real_index]

        model = PoseRAC(None, None, None, None, dim=99, heads=9, enc_layer=6, learning_rate=0.001, seed=42, num_classes=self.num_classes, alpha=0.012)
        weight_path = '..\\best_weights_PoseRAC.pth'
        new_weights = torch.load(weight_path, map_location='cpu')
        model.load_state_dict(new_weights)
        model.eval()
        self.model = model

        self.repetition_salient_1 = Action_trigger(action_name=self.action_type, enter_threshold=0.78, exit_threshold=0.4)
        self.repetition_salient_2 = Action_trigger(action_name=self.action_type, enter_threshold=0.78, exit_threshold=0.4)
        self.pose_classification_visualizer = PoseClassificationVisualizer(class_name=self.action_type, plot_x_max=500, plot_y_max=1)

    def load_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Video", "", "Video Files (*.mp4 *.avi *.mkv *.mov)")
        if file_path:
            # 打开视频文件
            self.cap = cv2.VideoCapture(file_path)
            if self.cap.isOpened():
                # 开始定时器
                self.timer.start(30)  # 设置定时器间隔为30毫秒（约33帧每秒）
            else:
                QMessageBox.information(self, "警告", "无法打开视频文件", QMessageBox.Ok)

    def update_frame(self):
        if self.cap and self.cap.isOpened():
            ret, input_frame = self.cap.read()
            if ret:
                self.frame_count += 1
                input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
                result = mp_pose.Pose().process(image=input_frame)
                pose_landmarks = result.pose_landmarks

                output_frame = input_frame.copy()
                if pose_landmarks is not None:
                    mp_drawing.draw_landmarks(image=output_frame, landmark_list=pose_landmarks, connections=mp_pose.POSE_CONNECTIONS)

                    landmarks = get_landmarks(pose_landmarks)
                    normalized_landmarks = normalize_landmarks(landmarks)
                    normalized_landmarks = normalized_landmarks[0].reshape(-1, 99)
                    tensor = torch.from_numpy(normalized_landmarks).float()

                    if self.tensors is None:
                        self.tensors = tensor
                    else:
                        if self.tensors.shape[0] == 100:
                            self.tensors = self.tensors[1:]
                        self.tensors = torch.cat((self.tensors, tensor), 0)

                    if self.tensors.shape[0] < 30:
                        return

                    output = torch.sigmoid(self.model(self.tensors))[-1]
                    output_numpy = output[self.real_index].detach().cpu().numpy()
                    self.classify_prob = output_numpy * (1. - self.momentum) + self.momentum * self.classify_prob

                    salient1_triggered = self.repetition_salient_1(self.classify_prob)
                    reverse_classify_prob = 1 - self.classify_prob
                    salient2_triggered = self.repetition_salient_2(reverse_classify_prob)

                    if self.init_pose == 'pose_holder':
                        if salient1_triggered:
                            self.init_pose = 'salient1'
                        elif salient2_triggered:
                            self.init_pose = 'salient2'

                    if self.init_pose == 'salient1':
                        if self.curr_pose == 'salient1' and salient2_triggered:
                            self.pose_count += 1
                    else:
                        if self.curr_pose == 'salient2' and salient1_triggered:
                            self.pose_count += 1

                    if salient1_triggered:
                        self.curr_pose = 'salient1'
                    elif salient2_triggered:
                        self.curr_pose = 'salient2'

                    output_frame = self.pose_classification_visualizer(frame=output_frame, pose_classification=self.classify_prob, pose_classification_filtered=self.classify_prob, repetitions_count=self.pose_count)

                h, w, ch = output_frame.shape
                bytes_per_line = ch * w
                q_image = QImage(output_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_image)
                self.video_label.setPixmap(pixmap)

            else:
                self.cap.release()
                self.timer.stop()

    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        self.timer.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    player = VideoPlayer()
    player.show()
    sys.exit(app.exec_())

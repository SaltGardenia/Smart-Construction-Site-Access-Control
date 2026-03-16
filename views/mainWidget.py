import cv2
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import threading
import os
import json
import time


class mainWidget(QWidget):
    def __init__(self):
        super(mainWidget, self).__init__()
        self.setWindowTitle('智慧工地门禁系统')
        self.resize(900, 1200)
        self.setWindowIcon(QIcon('img/icon.png'))

        self.title_lable = QLabel('智慧工地门禁系统', self)  # 标题
        self.title_lable.setStyleSheet('''max-height: 60px;
                                       font-family: "Microsoft YaHei", 宋体;
                                       font-size: 22px;
                                       font-weight: bold''')
        self.title_lable.setAlignment(Qt.AlignCenter)

        # 按钮
        self.turn_on_the_camera_btn = QPushButton('打开摄像头')
        self.screen_shot_btn = QPushButton('截屏')
        self.collect_information_btn = QPushButton('采集信息')
        self.practice_btn = QPushButton('训练')
        self.test_btn = QPushButton('检测')

        self.turn_on_the_camera_btn.clicked.connect(self.start_camera)

        self.btn_layout = QHBoxLayout()
        for btn in [self.turn_on_the_camera_btn,
                    self.screen_shot_btn,
                    self.collect_information_btn,
                    self.practice_btn]:
            self.btn_layout.addWidget(btn)

        # 摄像头 - 设置固定高度
        self.camera_lable = QLabel()
        self.camera_lable.setAlignment(Qt.AlignCenter)  # 居中
        self.camera_lable.setMinimumSize(640, 480)  # 设置最小尺寸
        self.camera_lable.setMaximumHeight(600)  # 设置最大高度为600像素
        self.camera_lable.setStyleSheet("""
            background-color: black; 
            border: 1px solid gray;
            max-height: 600px;
        """)
        self.camera_lable.setText("摄像头未开启")
        self.camera_lable.setScaledContents(False)  # 不自动缩放内容

        self.cap = None  # 摄像头对象（初始为空）
        self.timer = QTimer(self)  # 定时器（用于循环读取帧）
        self.timer.timeout.connect(self.update_frame)  # 定时器触发时更新画面
        self.timer_interval = 30  # 刷新间隔（毫秒，30ms ≈ 33帧/秒）

        # 创建定时器用于更新时间
        self.time_timer = QTimer(self)
        self.time_timer.timeout.connect(self.update_time)
        self.time_timer.start(1000)  # 每秒更新一次

        # 系统提示 截屏预览
        self.system_tips_lable = QLabel()  # 系统提示
        # 初始化时显示当前时间
        self.update_time()
        self.system_tips_lable.setMaximumHeight(200)  # 限制系统提示区域高度

        self.screen_shot_preview_lable = QLabel()  # 截屏预览
        self.screen_shot_preview_lable.setText('截屏预览')
        self.screen_shot_preview_lable.setMaximumHeight(200)  # 限制截屏预览区域高度

        self.information_layout = QHBoxLayout()
        self.information_layout.addWidget(self.system_tips_lable)
        self.information_layout.addWidget(self.screen_shot_preview_lable)

        self.total_layout = QVBoxLayout()  # 总布局
        self.total_layout.addWidget(self.title_lable)  # 标题
        self.total_layout.addLayout(self.btn_layout)  # 按钮

        # 添加摄像头标签到布局，并设置拉伸因子
        self.total_layout.addWidget(self.camera_lable, 1)  # 第二个参数是拉伸因子

        self.total_layout.addLayout(self.information_layout)  # 信息： 系统提示 截屏预览
        self.setLayout(self.total_layout)  # 设置总布局

        self.apply_styles()

        # 提前初始化摄像头资源
        self.camera_initialized = False
        self.camera_lock = threading.Lock()
        threading.Thread(target=self.initialize_camera, daemon=True).start()

        # 级联分类器
        self.faceModel = cv2.CascadeClassifier('util/haarcascade_frontalface_default.xml')

    def update_time(self):
        """更新系统时间显示"""
        current_time = QDateTime.currentDateTime()
        time_text = "当前时间: " + current_time.toString("yyyy-MM-dd hh:mm:ss")
        self.system_tips_lable.setText(time_text)

    def initialize_camera(self):
        """后台线程初始化摄像头资源"""
        with self.camera_lock:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.camera_initialized = True

    def start_camera(self):
        """启动摄像头"""
        with self.camera_lock:
            if not self.camera_initialized:
                self.cap = cv2.VideoCapture(0)  # 重新初始化摄像头资源
                if not self.cap.isOpened():
                    QMessageBox.warning(self, "错误", "无法打开摄像头！")
                    return
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.camera_initialized = True

        # 启用定时器，开始刷新画面
        self.timer.start(self.timer_interval)
        # 更新按钮状态
        self.turn_on_the_camera_btn.setText('关闭摄像头')
        self.turn_on_the_camera_btn.clicked.disconnect()
        self.turn_on_the_camera_btn.clicked.connect(self.stop_camera)

    def stop_camera(self):
        """停止摄像头"""
        self.timer.stop()
        with self.camera_lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
        self.camera_initialized = False  # 确保标志位正确更新
        self.camera_lable.clear()
        self.camera_lable.setText("摄像头已关闭")
        self.turn_on_the_camera_btn.setText('打开摄像头')
        self.turn_on_the_camera_btn.clicked.disconnect()
        self.turn_on_the_camera_btn.clicked.connect(self.start_camera)

    def update_frame(self):
        """读取摄像头帧并更新到 QLabel"""
        if self.cap is None or not self.cap.isOpened():
            return

        # 1. 读取摄像头帧（ret：是否读取成功，frame：帧数据）
        ret, frame = self.cap.read()
        if not ret:
            print("警告：无法读取摄像头画面！")
            self.stop_camera()
            return

        # 2. 格式转换：OpenCV 读取的是 BGR 格式，PyQt 显示需要 RGB 格式
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 人脸检测
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.faceModel.detectMultiScale(gray_frame, 1.1, 3, minSize=(200, 200), maxSize=(300, 300))
        for x, y, w, h in faces:
            cv2.rectangle(rgb_frame, (x, y), (x + w, y + h), (0, 255, 0), 2, cv2.LINE_AA)

        # 3. 获取摄像头标签的当前大小（考虑最大高度限制）
        label_size = self.camera_lable.size()

        # 4. 计算保持宽高比的缩放尺寸
        h, w = frame.shape[:2]
        aspect_ratio = w / h

        # 根据最大高度计算宽度
        max_height = min(label_size.height(), 600)  # 不超过600px
        new_height = max_height
        new_width = int(new_height * aspect_ratio)

        # 如果计算出的宽度超过标签宽度，则按宽度缩放
        if new_width > label_size.width():
            new_width = label_size.width()
            new_height = int(new_width / aspect_ratio)

        # 5. 调整画面大小
        resized_frame = cv2.resize(rgb_frame, (new_width, new_height),
                                   interpolation=cv2.INTER_AREA)

        # 6. 转换为 QImage（PyQt 可显示的图像格式）
        h, w, ch = resized_frame.shape
        bytes_per_line = ch * w
        q_image = QImage(resized_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # 7. 转换为 QPixmap 并显示到 QLabel
        self.camera_lable.setPixmap(QPixmap.fromImage(q_image))

    def closeEvent(self, event):
        """窗口关闭时释放摄像头资源"""
        self.stop_camera()
        # 停止时间定时器
        self.time_timer.stop()
        event.accept()

    def apply_styles(self):
        """应用样式"""
        self.setStyleSheet("""
                QWidget {
                    background-color: white;
                }
                QPushButton {
                    background-color: #4CAF50;
                    border: none;
                    color: white;
                    padding: 8px 16px;
                    text-align: center;
                    font-size: 14px;
                    margin: 4px 2px;
                    border-radius: 8px;
                }
                QLabel {
                    background-color: rgba(255, 255, 255);
                    border-radius: 20px;
                    padding: 10px;
                }
                #camera_lable {
                    max-height: 600px;
                    background-color: black;
                    border: 2px solid gray;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
                QPushButton:pressed {
                    background-color: #3d8b40;
                }
                QTextEdit {
                    border: 1px solid #CCCCCC;
                    border-radius: 3px;
                    padding: 5px;
                    background-color: #FAFAFA;
                }
            """)

        # 为摄像头标签设置对象名称，便于CSS选择器精确控制
        self.camera_lable.setObjectName("camera_lable")


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = mainWidget()
    window.show()
    sys.exit(app.exec_())
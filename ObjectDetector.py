from kivymd.app import MDApp
from kivy.lang import Builder
from kivy.core.window import Window
import numpy as np
import cv2
import os

Window.size = (400, 600)

kv = """
<MyTile@SmartTileWithStar>
    size_hint_y: None
    height: "180dp"
    
ScrollView:
    BoxLayout:
        orientation: "vertical"
        MDToolbar:
            title: "Object Detector App"
    
        MDLabel:
            text: "" 
    
        MDLabel:
            text: "Welcome User!"
            font_size: 17
            halign: "center"
            pos_hint: {'center_y': 0.2}
            theme_text_color: "Secondary" 
            
        MDGridLayout:
            cols: 1
            adaptive_height: True
            padding: dp(70), dp(70)
            spacing: dp(40)
    
            MyTile:
                source: "Camera.png"
                text: "[size=15]Open Camera to Detect now[/size]"
                on_release: app.showcam() 
                
        MDLabel:
            text: "Here you can now detect objects using your own camera!" 
            halign: "center"
            theme_text_color: "Secondary" 
            
        MDLabel:
            text: ""
            
        MDBottomNavigation:
            MDBottomNavigationItem:
                name: 'screen 1'
                text: 'Home'
                icon: 'home'
            MDBottomNavigationItem:
                name: 'screen 2'
                text: 'Exit'
                icon: 'exit-to-app'
                on_tab_release: app.exit() 
                        
"""
class ObjDetector(MDApp):

    def build(self):
        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "Teal"
        run = Builder.load_string(kv)
        return run

    def showcam(self):
        thres = 0.5  # Threshold to detect object
        nms_threshold = 0.2  # (0.1 to 1) 1 means no suppress , 0.1 means high suppress
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 400)  # width
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)  # height
        cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)  # brightness

        classNames = []
        with open('coco.names', 'r') as f:
            classNames = f.read().splitlines()

        font = cv2.FONT_HERSHEY_PLAIN

        Colors = np.random.uniform(0, 255, size=(len(classNames), 3))

        weightsPath = "frozen_inference_graph.pb"
        configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
        net = cv2.dnn_DetectionModel(weightsPath, configPath)
        net.setInputSize(320, 320)
        net.setInputScale(1.0 / 127.5)
        net.setInputMean((127.5, 127.5, 127.5))
        net.setInputSwapRB(True)

        while True:
            success, img = cap.read()
            classIds, confs, bbox = net.detect(img, confThreshold=thres)
            bbox = list(bbox)
            confs = list(np.array(confs).reshape(1, -1)[0])
            confs = list(map(float, confs))

            #print(type(confs[0]))

            indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)
            key = cv2.waitKey(20)
            cv2.imshow("Output", img)
            cv2.waitKey(1)
            if len(classIds) != 0:
                for i in indices:
                    i = i[0]
                    box = bbox[i]
                    confidence = str(round(confs[i], 2))
                    color = Colors[classIds[i][0] - 1]
                    x, y, w, h = box[0], box[1], box[2], box[3]
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness=2)
                    cv2.putText(img, classNames[classIds[i][0] - 1] + " " + confidence, (x + 10, y + 20),
                                font, 1, color, 2)
            #           cv2.putText(img,str(round(confidence,2)),(box[0]+100,box[1]+30),
            #                         font,1,colors[classId-1],2)
                    cv2.imshow("Output", img)
                    cv2.waitKey(1)

                if key == 27:
                    cv2.destroyWindow("Output")
                    break


    def exit(self):
        MDApp.get_running_app().stop()


ObjDetector().run()
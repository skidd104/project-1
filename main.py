from kivymd.app import MDApp
from kivymd.uix.screen import Screen
from kivymd.uix.button import MDRaisedButton
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.camera import Camera
from kivy.core.window import Window
from kivy.clock import Clock
import numpy as np
import cv2
from ultralytics import YOLO
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
import pyttsx3
import gc

class VideoCaptureApp(MDApp):
    def build(self):
        self.theme_cls.primary_palette = "Teal"
        self.theme_cls.primary_hue = "500"
        Window.size = (500, 700)

        self.screen = Screen()
        layout = BoxLayout(orientation="vertical", spacing=10, padding=10)

        self.camera = Camera(play=False, resolution=(640, 480))
        self.image = Image()
        layout.add_widget(self.image)

        btn_start_capture = MDRaisedButton(
            text="Start Video Capture", on_release=self.start_video_capture
        )

        btn_stop_capture = MDRaisedButton(
            text="Stop Video Capture", on_release=self.stop_video_capture
        )


        layout.add_widget(btn_start_capture)
        layout.add_widget(btn_stop_capture)

        self.screen.add_widget(layout)
        return self.screen

    def start_video_capture(self, instance):
        self.camera.play = not self.camera.play
        if self.camera.play:
            Clock.schedule_interval(self.capture_frame, 0.1)
    
    def stop_video_capture(self, instance):
        self.camera.play = False
        Clock.unschedule(self.capture_frame)
        
    

    def capture_frame(self, dt):
        frame = self.camera.texture
        frame = np.frombuffer(frame.pixels, dtype=np.uint8)
        frame = frame.reshape(self.camera.texture.height, self.camera.texture.width, 4)[:, :, :3]


        frame_c1 = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

        
        #results = model.predict (frame_c1, show=False, conf=0.45)
        results = model (frame_c1, show=False, conf=0.45)
        frame_copy = frame_c1.copy()
        
        
        for box in results[0].boxes.xyxy:
            x, y, w, h = map(int, box[:4])
            scale_factor = 0.5
            w = int (w * scale_factor)
            h = int (h * 1.0)

            cv2.rectangle(frame_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cls_tensor = results[0].boxes.cls

            if cls_tensor.numel() > 0:
                cls_value = cls_tensor.item ()
                print ("CLS VALUE:", cls_value)
                if int(cls_value) == 0:
                    print ("GREEN")
                    engine.say("GREEN")
                    engine.runAndWait()
                    class_label = "Green"
                elif int(cls_value) == 1:
                    print ("RED")
                    engine.say("RED")
                    engine.runAndWait()
                    class_label = "Red"
                elif int(cls_value) == 2:
                    print ("YELLOW")
                    engine.say("YELLOW")
                    engine.runAndWait()
                    class_label = "Yellow"
                
                label = f"{class_label}: {cls_value}"
                cv2.putText (frame_copy, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                print ("No elements")
        


        frame_copy = cv2.flip(frame_copy, 0)
        
        frame_rgb = cv2.cvtColor (frame_copy, cv2.COLOR_BGR2RGB)

        texture = Texture.create (size=(frame_copy.shape[1], frame_copy.shape[0]), colorfmt='rgb')
        texture.blit_buffer(frame_rgb.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
        self.image.texture = texture
        del results
        gc.collect()


if __name__ == "__main__":
    model = YOLO("best.pt")
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    VideoCaptureApp().run()

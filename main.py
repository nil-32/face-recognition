import concurrent.futures
import datetime
import tkinter as tk
from tkinter import ttk
import face_recognition
import sys, os
import cv2
import numpy as np
from PIL import Image, ImageTk

class FaceRecognition:

    face_locations =[]
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True
    data = {}

    def __init__(self):
        self.encode_faces()


    def encode_faces(self):

        for image in os.listdir('my_db'):
            face_image = face_recognition.load_image_file(f'my_db/{image}')
            face_encoding = face_recognition.face_encodings(face_image)[0]

            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(image)
            self.data.update({image: {"name":image.split(".")[0],"last attendance time" : "non"}})
        print(self.data)
        #print(self.known_face_names)


    def run_recognition(self):
        root = tk.Tk()
        root.geometry("1100x500")
        label_widget = ttk.Label(root)
        table = ttk.Treeview(root, columns=('name', 'last attendance'), show='headings')

        table.column("# 1")
        table.heading("# 1", text="name")
        table.column("# 2", anchor=tk.NW)
        table.heading("# 2", text="last attendance time")
        root.columnconfigure(0,weight=1)
        root.columnconfigure(1, weight=1)
        for key in self.data.keys():
            table.insert(parent='',text= "key",index= tk.END ,values= (self.data[key]['name'], self.data[key]['last attendance time']))
        table.column("#0", width=0, stretch=False)

        label_widget.grid( column=0 , padx=5, pady=5)
        table.grid(column= 1,row=0)
        vid = cv2.VideoCapture(0)

        if not vid.isOpened():
            sys.exit("no source")

        while True:
            ret, frame = vid.read()

            if self.process_current_frame:
                small_frame = cv2.resize(frame, (0,0),fx=0.25,fy=0.25)

                self.face_locations = face_recognition.face_locations(small_frame)
                self.face_encodings = face_recognition.face_encodings(face_image=small_frame,known_face_locations= self.face_locations)

                self.face_names = []

                def match_faces(face_encoding):
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = 'unknown'
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)

                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                    self.face_names.append(f'{name.split(".")[0]}')
                    dt_string = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                    self.data[name] = {}
                    self.data[name]["last attendence time"] = dt_string
                    print(self.data)
                    '''if name != "unknown":
                        panelC = tk.Label(image=cv2.imread(f"./my_db/{name}"))'''
                    table.insert(parent='', text="key", index=tk.END, values=(self.data[name]['name'], dt_string))
                    table.grid(column= 1,row=0)
                    table.configure()
                    table.update()
                    root.update()

                with concurrent.futures.ThreadPoolExecutor() as executor :
                    executor.map(match_faces,self.face_encodings)

            self.process_current_frame = not self.process_current_frame

            for (t,r,b,l), name in zip(self.face_locations,self.face_names):
                t *= 4
                r *= 4
                b *= 4
                l *= 4

                cv2.rectangle(frame, (l,t),(r,b),(0,0,255),2)
                cv2.rectangle(frame, (l,b-35),(r,b),(0,0,255),-1)
                cv2.putText(frame,name, (l+6,b-6),cv2.FONT_HERSHEY_DUPLEX,0.8,(255,255,255),1)

            #cv2.imshow('FAce Recognition',frame)
            opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            captured_image = Image.fromarray(opencv_image)
            photo_image = ImageTk.PhotoImage(image=captured_image)
            label_widget.configure(image=photo_image)

            root.update()
            if cv2.waitKey(1) == ord('q'):
                break
        root.mainloop()
        vid.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    fr = FaceRecognition()
    fr.run_recognition()
    print(fr.data)

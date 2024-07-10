import face_recognition
import sys, os
import cv2
import numpy as np

class FaceRecognition:

    face_locations =[]
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True

    def __init__(self):
        self.encode_faces()

    def encode_faces(self):

        for image in os.listdir('my_db'):
            face_image = face_recognition.load_image_file(f'my_db/{image}')
            face_encoding = face_recognition.face_encodings(face_image)[0]

            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(image)

        print(self.known_face_names)
    def run_recognition(self):

        vid = cv2.VideoCapture(0)

        if not vid.isOpened():
            sys.exit("no source")

        while True:
            ret, frame = vid.read()

            if self.process_current_frame:
                small_frame = cv2.resize(frame, (0,0),fx=0.25,fy=0.25)
                rgb_small_frame = small_frame[:, :, ::-1]

                self.face_locations = face_recognition.face_locations(rgb_small_frame)
                #p1img = face_recognition.load_image_file(rgb_small_frame)
                self.face_encodings = face_recognition.face_encodings(face_image=small_frame,known_face_locations= self.face_locations)

                self.face_names = []
                for face_encoding in self.face_encodings:
                    matches = face_recognition.compare_faces(self.known_face_encodings,face_encoding)
                    name= 'unknown'
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)

                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                    self.face_names.append(f'{name.split(".")[0]}')
            self.process_current_frame = not self.process_current_frame

            for (t,r,b,l), name in zip(self.face_locations,self.face_names):
                t *= 4
                r *= 4
                b *= 4
                l *= 4

                cv2.rectangle(frame, (l,t),(r,b),(0,0,255),2)
                cv2.rectangle(frame, (l,b-35),(r,b),(0,0,255),-1)
                cv2.putText(frame,name, (l+6,b-6),cv2.FONT_HERSHEY_DUPLEX,0.8,(255,255,255),1)

            cv2.imshow('FAce Recognition',frame)
            if cv2.waitKey(1) == ord('q'):
                break

        vid.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    fr = FaceRecognition()
    fr.run_recognition()

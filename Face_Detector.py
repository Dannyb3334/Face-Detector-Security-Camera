import cv2, random, keyboard
#Load pre-trained data on frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Capture video from webcam
webcam = cv2.VideoCapture(0)

#Iterate through all frames of video feed
while True:

    #Read the current frame
    successful_frame_read, frame = webcam.read()

    #Convert image to grayscale (Must)
    grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #Detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_frame)

    #Draw rectangles around detected faces (img, (x,y), (x+w, y+h), ( B, G, R), border weight)
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 256, 0), 2)

    #Print face coordinates
    print(face_coordinates)

    #Display image with detected face
    cv2.imshow('Face Detector', frame)
    cv2.waitKey(2)
    #When "space" is held end video
    if keyboard.is_pressed('space'):
        break
    else:
        continue
print("Code Completed")
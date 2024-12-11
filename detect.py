import cv2
import mediapipe as mp
import numpy as np

# Inisialisasi Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Fungsi untuk menghitung jumlah jari yang terangkat
def count_fingers(hand_landmarks):
    fingers = [False] * 5  # Jempol, telunjuk, tengah, manis, kelingking
    
    # Landmark jari
    tips = [4, 8, 12, 16, 20]
    
    # Cek apakah jempol terangkat
    if hand_landmarks.landmark[tips[0]].x < hand_landmarks.landmark[tips[0] - 1].x:  # Jempol (x-axis)
        fingers[0] = True
    
    # Cek jari lainnya
    for i in range(1, 5):  # Telunjuk hingga kelingking (y-axis)
        if hand_landmarks.landmark[tips[i]].y < hand_landmarks.landmark[tips[i] - 2].y:
            fingers[i] = True
    
    return fingers.count(True)

# Fungsi untuk mengenali gestur
def recognize_gesture(finger_count):
    if finger_count == 5:
        return "Stop"
    elif finger_count == 2:
        return "Peace"
    elif finger_count == 1:
        return "Like"
    elif finger_count == 0:
        return "Fist"
    else:
        return "Unknown"

# Buka kamera
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5) as hands:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Konversi ke RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Deteksi tangan
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Gambar landmark dan hitung jumlah jari
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Hitung jumlah jari
                finger_count = count_fingers(hand_landmarks)
                
                # Kenali gestur
                gesture = recognize_gesture(finger_count)
                
                # Efek visual
                if gesture == "Peace":
                    cv2.putText(image, "PEACE!", (10, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 5)
                elif gesture == "Like":
                    cv2.circle(image, (200, 200), 100, (0, 255, 0), -1)
                elif gesture == "Stop":
                    cv2.rectangle(image, (50, 50), (300, 300), (255, 0, 0), 5)
                
                # Tampilkan jumlah jari dan gestur
                cv2.putText(image, f'Jari: {finger_count}', (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(image, f'Gestur: {gesture}', (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # Tampilkan hasil
        cv2.imshow('Deteksi Jumlah Jari & Gestur', image)
        
        # Keluar dengan menekan tombol 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Bersihkan
cap.release()
cv2.destroyAllWindows()
 
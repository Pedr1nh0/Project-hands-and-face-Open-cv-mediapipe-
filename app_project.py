import cv2
import mediapipe as mp
import numpy as np

# -------------------- CONFIGURAÇÃO --------------------
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils

# Índices dos landmarks
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [78, 81, 13, 311]  # simplificado

# -------------------- FUNÇÕES --------------------
def eye_aspect_ratio(landmarks, eye_points):
    pts = np.array([(landmarks[p].x, landmarks[p].y) for p in eye_points])
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3]) + 1e-9
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(landmarks):
    top = np.array([landmarks[13].x, landmarks[13].y])
    bottom = np.array([landmarks[14].x, landmarks[14].y])
    left = np.array([landmarks[78].x, landmarks[78].y])
    right = np.array([landmarks[308].x, landmarks[308].y])
    vertical = np.linalg.norm(top - bottom)
    horizontal = np.linalg.norm(left - right) + 1e-9
    return vertical / horizontal

# Limiares fixos simples
EAR_THRESH = 0.21  # olhos fechados se abaixo
MAR_THRESH = 0.5   # sorriso se acima

# -------------------- DETECÇÃO --------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)   # CORREÇÃO: flip no início

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # FACE
    face_results = face_mesh.process(frame_rgb)
    if face_results.multi_face_landmarks:
        landmarks = face_results.multi_face_landmarks[0].landmark

        ear = (eye_aspect_ratio(landmarks, LEFT_EYE) + eye_aspect_ratio(landmarks, RIGHT_EYE)) / 2
        mar = mouth_aspect_ratio(landmarks)

        olhos = "Fechados" if ear < EAR_THRESH else "Abertos"
        boca = "Boca aberta" if mar > MAR_THRESH else "Neutro"

        cv2.putText(frame, f"Olhos: {olhos}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
        cv2.putText(frame, f"Boca: {boca}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)

        # Desenha face landmarks
        mp_drawing.draw_landmarks(frame, face_results.multi_face_landmarks[0], mp_face_mesh.FACEMESH_TESSELATION,
                                  mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1),
                                  mp_drawing.DrawingSpec(color=(0,255,0), thickness=1))

    # HANDS
    hand_results = hands_detector.process(frame_rgb)
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Face + Mãos", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

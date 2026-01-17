import cv2
import mediapipe as mp
import numpy as np

# ---------------- SETUP ----------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS
CLONE_OFFSET = 350  # pixels to the right for the clone

# ---------------- HELPERS ----------------
def landmarks_to_points(landmarks, w, h):
    pts = []
    for lm in landmarks.landmark:
        pts.append((int(lm.x * w), int(lm.y * h)))
    return pts

def draw_skeleton(img, points, color, thickness=2):
    for a, b in POSE_CONNECTIONS:
        if a < len(points) and b < len(points):
            cv2.line(img, points[a], points[b], color, thickness)
    for x, y in points:
        cv2.circle(img, (x, y), 3, color, -1)

def add_neon_glow(base, draw_layer, glow_strength=2.0, blur_ksize=31):
    """Create glow by blurring the draw_layer and blending it back."""
    glow = cv2.GaussianBlur(draw_layer, (blur_ksize, blur_ksize), 0)
    return cv2.addWeighted(base, 1.0, glow, glow_strength, 0)

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    # Layers
    base = frame.copy()
    draw_layer = np.zeros_like(frame)

    if result.pose_landmarks:
        pts = landmarks_to_points(result.pose_landmarks, w, h)

        # Original skeleton (cyan)
        draw_skeleton(draw_layer, pts, (0, 255, 255), thickness=2)

        # Mirror clone (magenta), shifted to the right
        clone_pts = []
        for x, y in pts:
            clone_x = w - x + CLONE_OFFSET
            clone_pts.append((clone_x, y))
        draw_skeleton(draw_layer, clone_pts, (255, 0, 255), thickness=2)

    # Add neon glow
    output = add_neon_glow(base, draw_layer, glow_strength=1.8, blur_ksize=35)

    # Overlay crisp lines on top for sharp edges
    output = cv2.addWeighted(output, 1.0, draw_layer, 1.0, 0)

    cv2.putText(
        output,
        "Live Pose Mirror Clone • Neon Glow",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Pose Clone - Neon", output)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()

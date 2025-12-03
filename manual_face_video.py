import cv2
import mediapipe as mp
import numpy as np
import os

# Create directories for saving results
RESULT_DIR = '/Users/takiguchiryosei/Documents/face_recognition/excursion_results/result'
VIDEO_DIR = os.path.join(RESULT_DIR, 'video')
GRAPH_DIR = os.path.join(RESULT_DIR, 'graph')

os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)

# Update paths for saving results
VIDEO_PATH = os.path.join(VIDEO_DIR, 'result_video.mp4')
GRAPH_PATH = os.path.join(GRAPH_DIR, 'result_graph.png')

# Replace RESULT_PATH with GRAPH_PATH for saving graphs
RESULT_PATH = GRAPH_PATH

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5)

KEY_POINTS = {
    'Eyebrow': (105, 334),
    'Eye': (33, 263),   
    'InnerEye': (133, 362), 
    'Mouth': (61, 291),
}

def capture_from_webcam():
    """
    Webカメラを起動し、スペースキーで画像を撮影して返す関数
    """
    cap = cv2.VideoCapture(0) # カメラ起動
    if not cap.isOpened():
        print("カメラが見つかりませんでした。")
        return None

    print("--- 撮影モード ---")
    print(" [SPACE]: 撮影して決定")
    print(" [ESC]: 中止")

    captured_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("フレームの取得に失敗しました。")
            break

        # 鏡のように左右反転（自分を見る時に自然なように）
        frame = cv2.flip(frame, 1)

        # 画面にガイドを表示
        display_frame = frame.copy()
        h, w = display_frame.shape[:2]
        
        # 中央のガイド線
        cv2.line(display_frame, (w//2, 0), (w//2, h), (0, 255, 255), 1)
        cv2.putText(display_frame, "Press [SPACE] to Capture", (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imshow('Webcam - Capture', display_frame)

        key = cv2.waitKey(1)
        if key == 32: # Space Key
            captured_frame = frame
            break
        elif key == 27: # ESC Key
            break

    cap.release()
    cv2.destroyAllWindows()
    return captured_frame

def apply_transform(image, angle, tx, ty):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    M[0, 2] += tx
    M[1, 2] += ty
    transformed = cv2.warpAffine(image, M, (w, h), borderValue=(200, 200, 200))
    return transformed

def manual_alignment_grid(image):
    """
    太い線は縦のみ＋横は等間隔グリッド
    """
    angle = 0.0
    tx = 0
    ty = 0
    
    base_image = image.copy()
    h, w = base_image.shape[:2]
    cx, cy = w // 2, h // 2 
    
    print("--- アライメントモード ---")
    print(" [矢印キー]: 移動  [Z/X]: 回転")
    print(" [Enter]: 確定")
    
    while True:
        display_img = apply_transform(base_image, angle, tx, ty)
        
        # --- グリッド描画 ---
        main_color = (0, 255, 255) # 黄色（太い）
        sub_color = (0, 150, 150)  # 暗めの黄色（細い）
        
        # 1. メインの縦線 (これだけ太く表示)
        cv2.line(display_img, (cx, 0), (cx, h), main_color, 2)
        
        # 2. 等間隔の水平ライン (位置合わせ用のガイド)
        step = 50 
        # 下方向
        for y in range(cy, h, step):
            cv2.line(display_img, (0, y), (w, y), sub_color, 1)
        # 上方向
        for y in range(cy, -1, -step):
            cv2.line(display_img, (0, y), (w, y), sub_color, 1)

        # 情報表示
        info = f"Angle: {angle:.1f}  X:{tx} Y:{ty}"
        cv2.putText(display_img, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
        
        cv2.imshow('Manual Alignment', display_img)
        
        key = cv2.waitKey(0)
        
        if key == 13: # Enter
            break
        elif key == 27: # ESC
            return None
            
        step_move = 2
        step_angle = 0.2
        
        if key == ord('z'): angle += step_angle
        elif key == ord('x'): angle -= step_angle
        elif key == 81 or key == 2424832: tx -= step_move 
        elif key == 82 or key == 2555904: ty -= step_move 
        elif key == 83 or key == 2621440: tx += step_move 
        elif key == 84 or key == 2490368: ty += step_move 
        elif key == 0: ty -= step_move
        elif key == 1: ty += step_move
        elif key == 2: tx -= step_move
        elif key == 3: tx += step_move

    cv2.destroyAllWindows()
    return apply_transform(base_image, angle, tx, ty)

def analyze_symmetry(image):
    h, w, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    if not results.multi_face_landmarks:
        print("顔検出失敗")
        return image

    landmarks = results.multi_face_landmarks[0].landmark
    l_eye = landmarks[KEY_POINTS['Eye'][0]].x * w
    r_eye = landmarks[KEY_POINTS['Eye'][1]].x * w
    face_width = abs(l_eye - r_eye)

    print("\n--- Symmetry Analysis Result ---")
    for name, (l_idx, r_idx) in KEY_POINTS.items():
        if name == 'InnerEye': continue 

        l_pt = landmarks[l_idx]
        r_pt = landmarks[r_idx]
        lp = np.array([l_pt.x * w, l_pt.y * h])
        rp = np.array([r_pt.x * w, r_pt.y * h])

        y_diff = lp[1] - rp[1]
        score = (y_diff / face_width) * 100

        print(f"{name.ljust(8)}: diff={y_diff:5.1f} px, score={score:5.1f}%")

        color = (0, 255, 0)
        if abs(score) > 2.0: color = (0, 0, 255)

        cv2.circle(image, tuple(lp.astype(int)), 4, color, -1)
        cv2.circle(image, tuple(rp.astype(int)), 4, color, -1)
        cv2.line(image, tuple(lp.astype(int)), tuple(rp.astype(int)), (200, 200, 200), 1)

        text = f"{score:.1f}%"
        cv2.putText(image, text, (int(lp[0])-30, int(lp[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 3)
        cv2.putText(image, text, (int(lp[0])-30, int(lp[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

    return image

# --- Main ---
# 1. ウェブカメラから画像を撮影
original_img = capture_from_webcam()

if original_img is not None:
    # 2. 手動で位置合わせ
    aligned_img = manual_alignment_grid(original_img)
    
    if aligned_img is not None:
        # 3. 解析して保存
        result_img = analyze_symmetry(aligned_img)
        cv2.imwrite(RESULT_PATH, result_img)
        
        print(f'\nResult saved to {RESULT_PATH}')
        
        # 結果を表示して確認（何かキーを押すと終了）
        cv2.imshow('Analysis Result', result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
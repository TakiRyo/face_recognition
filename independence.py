import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# ==========================================
# 設定: 保存先フォルダとファイル名
OUTPUT_DIR = '/Users/takiguchiryosei/Documents/face_recognition/independence_results'
VIDEO_NAME = 'independence_video_ishida.mp4'
GRAPH_NAME = 'independence_graph_ishida.png'
# ==========================================

# フォルダがなければ作成
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_VIDEO_PATH = os.path.join(OUTPUT_DIR, VIDEO_NAME)
OUTPUT_GRAPH_PATH = os.path.join(OUTPUT_DIR, GRAPH_NAME)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# --- ランドマークID定義 ---
ID_NOSE = 168
ID_MOUTH_L = 61
ID_MOUTH_R = 291
ID_FACE_W_L = 234
ID_FACE_W_R = 454

# EAR計算用
EYE_L_IDXS = [33, 160, 158, 133, 153, 144]
EYE_R_IDXS = [362, 385, 387, 263, 373, 380]

def calc_dist(p1, p2, w, h):
    return np.sqrt((p1.x*w - p2.x*w)**2 + (p1.y*h - p2.y*h)**2)

def calculate_ear(landmarks, indices, w, h):
    pts = [landmarks[i] for i in indices]
    v1 = calc_dist(pts[1], pts[5], w, h)
    v2 = calc_dist(pts[2], pts[4], w, h)
    h_dist = calc_dist(pts[0], pts[3], w, h)
    if h_dist == 0: return 0.0
    ear = (v1 + v2) / (2.0 * h_dist)
    return ear

def record_and_analyze_independence():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("カメラが見つかりません")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (w, h))

    frames_data = []
    is_recording = False
    
    print("\n=== 独立性 (Synkinesis) テスト ===")
    print(f"保存先: {OUTPUT_DIR}")
    print(" [SPACE]: 録画開始 (5秒間)")
    print(" 指示: 「目はパッチリ開けたまま、口だけ『イー』としてください」\n")

    start_time = 0
    
    while True:
        success, image = cap.read()
        if not success: break
        image = cv2.flip(image, 1)
        display_img = image.copy()
        
        if is_recording:
            elapsed = time.time() - start_time
            cv2.putText(display_img, f"REC: {elapsed:.1f}s", (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.circle(display_img, (w-40, 40), 10, (0, 0, 255), -1)

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            
            data_point = {'time': elapsed, 'valid': False}

            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                
                # 口座標
                p_nose = lm[ID_NOSE]
                p_mL = lm[ID_MOUTH_L]
                p_mR = lm[ID_MOUTH_R]
                fw = calc_dist(lm[ID_FACE_W_L], lm[ID_FACE_W_R], w, h)
                dist_mL = calc_dist(p_nose, p_mL, w, h) / fw
                dist_mR = calc_dist(p_nose, p_mR, w, h) / fw
                
                # EAR計算
                ear_L = calculate_ear(lm, EYE_L_IDXS, w, h)
                ear_R = calculate_ear(lm, EYE_R_IDXS, w, h)
                
                data_point.update({
                    'mouth_L': dist_mL, 'mouth_R': dist_mR,
                    'ear_L': ear_L, 'ear_R': ear_R,
                    'valid': True
                })

                # 描画
                for idx in EYE_L_IDXS + EYE_R_IDXS:
                    pt = (int(lm[idx].x*w), int(lm[idx].y*h))
                    cv2.circle(image, pt, 1, (0, 255, 255), -1)
                cv2.circle(image, (int(p_mL.x*w), int(p_mL.y*h)), 3, (0, 255, 0), -1)
                cv2.circle(image, (int(p_mR.x*w), int(p_mR.y*h)), 3, (0, 255, 0), -1)

            frames_data.append(data_point)
            writer.write(image)

            if elapsed > 5.0:
                print("録画終了。解析中...")
                break
        else:
            cv2.putText(display_img, "Press [SPACE] to Start", (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_img, "Move Mouth ONLY (Keep Eyes Open)", (20, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow('Independence Test', display_img)
        if cv2.waitKey(5) == 32 and not is_recording:
            is_recording = True
            start_time = time.time()
        elif cv2.waitKey(1) == 27:
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    
    if frames_data:
        analyze_independence_data(frames_data)

def analyze_independence_data(data):
    # ★変更点1: 閾値を30%に緩和
    THRESHOLD_PERCENT = 30.0 

    valid_data = [d for d in data if d['valid']]
    if not valid_data: return

    t = [d['time'] for d in valid_data]
    
    # --- データ整形 ---
    # ★変更点2: ベースラインを「平均」ではなく「最大値(or中央値)」にする
    # 最初の10フレームを見て、最も目が開いていた瞬間を基準(0%)とする
    # これにより、開始時の瞬きや半目が基準になるのを防ぐ
    initial_frames = 10
    if len(valid_data) < initial_frames: initial_frames = len(valid_data)

    base_ear_L = np.max([d['ear_L'] for d in valid_data[:initial_frames]])
    base_ear_R = np.max([d['ear_R'] for d in valid_data[:initial_frames]])
    
    # 口は中央値でOK
    base_mL = np.median([d['mouth_L'] for d in valid_data[:initial_frames]])
    base_mR = np.median([d['mouth_R'] for d in valid_data[:initial_frames]])
    
    # 口の変化 (%)
    move_L = [(base_mL - d['mouth_L']) * 100 for d in valid_data]
    move_R = [(base_mR - d['mouth_R']) * 100 for d in valid_data]
    
    # 目の変化 (%) - 移動平均で少しスムージングするとより良いが今回はシンプルに
    close_L = [(base_ear_L - d['ear_L']) / base_ear_L * 100 for d in valid_data]
    close_R = [(base_ear_R - d['ear_R']) / base_ear_R * 100 for d in valid_data]

    # --- グラフ描画 ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    ax1.plot(t, move_L, label='Left Mouth', color='blue')
    ax1.plot(t, move_R, label='Right Mouth', color='orange')
    ax1.set_title('Voluntary Movement: Mouth Excursion')
    ax1.set_ylabel('Mouth Lift (%)')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(t, close_L, label='Left Eye Closure', color='blue', linestyle='--')
    ax2.plot(t, close_R, label='Right Eye Closure', color='orange', linestyle='--')
    ax2.set_title('Involuntary Movement: Eye Closure (Synkinesis)')
    ax2.set_ylabel('Eye Closure (%)')
    ax2.set_xlabel('Time (sec)')
    
    # 判定ライン
    ax2.axhline(y=THRESHOLD_PERCENT, color='r', linestyle=':', alpha=0.5, label=f'Threshold ({THRESHOLD_PERCENT}%)')
    ax2.legend()
    ax2.grid(True)
    
    # --- 診断 ---
    # ノイズ対策: 一瞬だけのスパイクを除去するため、95パーセンタイル値などを評価に使う
    # ここではシンプルにMaxをとるが、閾値を上げたので大丈夫なはず
    max_syn_L = np.max(close_L)
    max_syn_R = np.max(close_R)
    
    print("\n" + "="*30)
    print(f"Max Eye Closure (Left) : {max_syn_L:.1f}%")
    print(f"Max Eye Closure (Right): {max_syn_R:.1f}%")
    print(f"Threshold: {THRESHOLD_PERCENT}%")
    
    if max_syn_L > THRESHOLD_PERCENT or max_syn_R > THRESHOLD_PERCENT:
        print("RESULT: Synkinesis Detected (共同運動の疑いあり)")
        side = "Left" if max_syn_L > max_syn_R else "Right"
        print(f"-> {side} side eye closed significantly.")
    else:
        print("RESULT: Good Independence (独立性は良好です)")
        print("-> No significant eye closure detected.")
    print("="*30 + "\n")

    plt.tight_layout()
    plt.savefig(OUTPUT_GRAPH_PATH)
    print(f"Graph saved to: {OUTPUT_GRAPH_PATH}")
    plt.show()

if __name__ == "__main__":
    record_and_analyze_independence()
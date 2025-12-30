# # # # ## your work ishida!
# # # # import cv2
# # # # import mediapipe as mp
# # # # import numpy as np
# # # # import matplotlib.pyplot as plt
# # # # import time
# # # # import os
# # # # import math

# # # # # ==========================================
# # # # # 設定エリア
# # # # # ==========================================
# # # # # 保存ディレクトリ（環境に合わせて変更してください）
# # # # RESULT_DIR = '/Users/takahiro/Documents/face_recognition/taki'
# # # # # RESULT_DIR = './results' # テスト用

# # # # # ファイル名設定
# # # # VIDEO_PATH = os.path.join(RESULT_DIR, 'vector_detail_video.mp4')
# # # # GRAPH_PATH_TRAJECTORY = os.path.join(RESULT_DIR, 'vector_trajectory.png') # 軌跡グラフ
# # # # GRAPH_PATH_COMPONENTS = os.path.join(RESULT_DIR, 'vector_components.png') # 成分分解グラフ

# # # # # ディレクトリ生成
# # # # os.makedirs(RESULT_DIR, exist_ok=True)

# # # # # MediaPipe設定
# # # # mp_face_mesh = mp.solutions.face_mesh
# # # # face_mesh = mp_face_mesh.FaceMesh(
# # # #     max_num_faces=1,
# # # #     refine_landmarks=True,
# # # #     min_detection_confidence=0.5,
# # # #     min_tracking_confidence=0.5)

# # # # # ランドマークID
# # # # ID_EYE_L = 33
# # # # ID_EYE_R = 263
# # # # ID_NOSE_BRIDGE = 168 
# # # # ID_MOUTH_L = 61
# # # # ID_MOUTH_R = 291

# # # # def get_coords(landmarks, idx, w, h):
# # # #     """ランドマークIDから(x, y)座標を取得"""
# # # #     return np.array([landmarks[idx].x * w, landmarks[idx].y * h])

# # # # def rotate_point(point, pivot, angle_rad):
# # # #     """
# # # #     点をピボットを中心に回転させる
# # # #     point: 回転させたい点 (x, y)
# # # #     pivot: 回転の中心 (x, y) -> 鼻を使用
# # # #     angle_rad: 回転角（ラジアン）
# # # #     """
# # # #     px, py = point
# # # #     cx, cy = pivot
    
# # # #     # 平行移動（中心を原点へ）
# # # #     tx, ty = px - cx, py - cy
    
# # # #     # 回転行列の適用
# # # #     cos_a = np.cos(angle_rad)
# # # #     sin_a = np.sin(angle_rad)
    
# # # #     # 回転後の座標
# # # #     nx = tx * cos_a - ty * sin_a
# # # #     ny = tx * sin_a + ty * cos_a
    
# # # #     # 平行移動（元の位置へ戻す ※今回は相対座標だけでいいので戻さなくても良いが、直感的に戻す）
# # # #     return np.array([nx + cx, ny + cy])

# # # # def record_and_analyze():
# # # #     cap = cv2.VideoCapture(0)
# # # #     if not cap.isOpened():
# # # #         print("カメラが見つかりません")
# # # #         return

# # # #     # 動画保存設定
# # # #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# # # #     fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
# # # #     w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# # # #     h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# # # #     writer = cv2.VideoWriter(VIDEO_PATH, fourcc, fps, (w, h))

# # # #     # データ格納用
# # # #     frames_data = [] 
    
# # # #     print("--- ベクトル詳細分析モード ---")
# # # #     print(" [SPACE]: 録画開始（真顔 -> 笑顔）")
# # # #     print(" 頭が傾いていても自動補正して計算します。")

# # # #     is_recording = False
# # # #     start_time = 0
    
# # # #     # 基準位置（最初の数フレームの平均）
# # # #     baseline_l = None
# # # #     baseline_r = None
# # # #     calibration_frames = []
# # # #     CALIBRATION_COUNT = 5 # 最初の5フレームを基準にする

# # # #     while True:
# # # #         success, image = cap.read()
# # # #         if not success: break

# # # #         image = cv2.flip(image, 1)
# # # #         display_image = image.copy()
# # # #         rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# # # #         results = face_mesh.process(rgb_image)

# # # #         if results.multi_face_landmarks:
# # # #             lm = results.multi_face_landmarks[0].landmark
            
# # # #             # --- 座標取得 ---
# # # #             p_eye_l = get_coords(lm, ID_EYE_L, w, h)
# # # #             p_eye_r = get_coords(lm, ID_EYE_R, w, h)
# # # #             p_nose = get_coords(lm, ID_NOSE_BRIDGE, w, h)
# # # #             p_mouth_l = get_coords(lm, ID_MOUTH_L, w, h)
# # # #             p_mouth_r = get_coords(lm, ID_MOUTH_R, w, h)

# # # #             # --- 顔の傾き計算 (水平補正用) ---
# # # #             # 右目と左目の角度を計算 (本来水平であるべきライン)
# # # #             delta_eyes = p_eye_r - p_eye_l
# # # #             face_angle = np.arctan2(delta_eyes[1], delta_eyes[0]) # ラジアン
            
# # # #             # --- 補正後の座標を計算 ---
# # # #             # 顔の傾きをキャンセルするように逆回転(-face_angle)させる
# # # #             # これにより「顔にとっての真上・真横」が座標軸と一致する
# # # #             fixed_mouth_l = rotate_point(p_mouth_l, p_nose, -face_angle)
# # # #             fixed_mouth_r = rotate_point(p_mouth_r, p_nose, -face_angle)
            
# # # #             # 正規化係数（顔幅）
# # # #             face_width = np.linalg.norm(p_eye_r - p_eye_l)

# # # #             # 描画（元の画像に）
# # # #             cv2.line(image, tuple(p_nose.astype(int)), tuple(p_mouth_l.astype(int)), (0, 255, 0), 1)
# # # #             cv2.line(image, tuple(p_nose.astype(int)), tuple(p_mouth_r.astype(int)), (0, 255, 0), 1)

# # # #             if is_recording:
# # # #                 elapsed = time.time() - start_time
                
# # # #                 # --- キャリブレーション（初期位置の特定）---
# # # #                 if len(calibration_frames) < CALIBRATION_COUNT:
# # # #                     calibration_frames.append({
# # # #                         'L': fixed_mouth_l,
# # # #                         'R': fixed_mouth_r
# # # #                     })
# # # #                     cv2.putText(display_image, "Calibrating...", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                
# # # #                 else:
# # # #                     # 基準点が決まったら計測開始
# # # #                     if baseline_l is None:
# # # #                         # 平均をとって基準点とする
# # # #                         l_points = np.array([f['L'] for f in calibration_frames])
# # # #                         r_points = np.array([f['R'] for f in calibration_frames])
# # # #                         baseline_l = np.mean(l_points, axis=0)
# # # #                         baseline_r = np.mean(r_points, axis=0)

# # # #                     # --- 変位ベクトルの計算 ---
# # # #                     # (現在位置 - 基準位置) / 顔幅
# # # #                     # Y軸は画像座標系では下がプラスなので、直感的にするために反転(-1倍)させて「上がプラス」にする
# # # #                     vec_l = (fixed_mouth_l - baseline_l) / face_width
# # # #                     vec_r = (fixed_mouth_r - baseline_r) / face_width
                    
# # # #                     # 保存データ: time, Left(dx, dy), Right(dx, dy)
# # # #                     # dx: 外側への広がり（左口角は左へ、右口角は右へ動くのが正）
# # # #                     # dy: 上方向への挙上
                    
# # # #                     # 左口角: xが減るのが「外側」なので符号反転して「外側移動量」にする
# # # #                     l_dx = -vec_l[0] 
# # # #                     l_dy = -vec_l[1] # 上方向をプラスに
                    
# # # #                     # 右口角: xが増えるのが「外側」
# # # #                     r_dx = vec_r[0]
# # # #                     r_dy = -vec_r[1] # 上方向をプラスに

# # # #                     frames_data.append({
# # # #                         'time': elapsed,
# # # #                         'L_dx': l_dx, 'L_dy': l_dy,
# # # #                         'R_dx': r_dx, 'R_dy': r_dy
# # # #                     })

# # # #                     cv2.putText(display_image, "REC", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# # # #                 if elapsed > 4.0:
# # # #                     break
            
# # # #             writer.write(image)

# # # #         cv2.imshow('Vector Analysis', display_image)
# # # #         key = cv2.waitKey(5)
# # # #         if key == 32 and not is_recording:
# # # #             is_recording = True
# # # #             start_time = time.time()
# # # #         elif key == 27:
# # # #             break

# # # #     cap.release()
# # # #     writer.release()
# # # #     cv2.destroyAllWindows()

# # # #     if frames_data:
# # # #         analyze_vectors(frames_data)

# # # # def analyze_vectors(data):
# # # #     # データを配列に変換
# # # #     times = [d['time'] for d in data]
# # # #     L_dx = [d['L_dx'] * 100 for d in data] # %表示
# # # #     L_dy = [d['L_dy'] * 100 for d in data]
# # # #     R_dx = [d['R_dx'] * 100 for d in data]
# # # #     R_dy = [d['R_dy'] * 100 for d in data]

# # # #     # --- グラフ1: 動きの成分分解 (時系列) ---
# # # #     fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
# # # #     # 上段: Vertical Lift (挙上)
# # # #     ax1.plot(times, L_dy, label='Left Lift (Vertical)', color='blue')
# # # #     ax1.plot(times, R_dy, label='Right Lift (Vertical)', color='orange')
# # # #     ax1.set_ylabel('Vertical Lift (% of Face Width)')
# # # #     ax1.set_title('Vertical Movement (How much it went UP)')
# # # #     ax1.grid(True)
# # # #     ax1.legend()

# # # #     # 下段: Horizontal Widen (拡大)
# # # #     ax2.plot(times, L_dx, label='Left Widen (Horizontal)', color='cyan', linestyle='--')
# # # #     ax2.plot(times, R_dx, label='Right Widen (Horizontal)', color='salmon', linestyle='--')
# # # #     ax2.set_ylabel('Horizontal Widen (% of Face Width)')
# # # #     ax2.set_xlabel('Time (sec)')
# # # #     ax2.set_title('Horizontal Movement (How much it went OUT)')
# # # #     ax2.grid(True)
# # # #     ax2.legend()
    
# # # #     plt.tight_layout()
# # # #     plt.savefig(GRAPH_PATH_COMPONENTS)
# # # #     print(f"成分分解グラフを保存しました: {GRAPH_PATH_COMPONENTS}")

# # # #     # --- グラフ2: ベクトル軌跡 (X vs Y) ---
# # # #     plt.figure(figsize=(8, 8))
    
# # # #     # 軌跡のプロット
# # # #     plt.plot(L_dx, L_dy, label='Left Trajectory', color='blue', marker='o', markersize=3, alpha=0.5)
# # # #     plt.plot(R_dx, R_dy, label='Right Trajectory', color='orange', marker='o', markersize=3, alpha=0.5)
    
# # # #     # 開始点と終了点
# # # #     plt.plot(L_dx[0], L_dy[0], 'bx', label='Start')
# # # #     plt.plot(R_dx[0], R_dy[0], 'rx', label='Start')
# # # #     # 最大挙上点
# # # #     max_idx_l = np.argmax(L_dy)
# # # #     max_idx_r = np.argmax(R_dy)
# # # #     plt.plot(L_dx[max_idx_l], L_dy[max_idx_l], 'bo', markersize=10, fillstyle='none', label='Max Lift')
# # # #     plt.plot(R_dx[max_idx_r], R_dy[max_idx_r], 'ro', markersize=10, fillstyle='none')

# # # #     # グラフ装飾
# # # #     plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
# # # #     plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)
# # # #     plt.title('Smile Vector Trajectory (Motion Path)')
# # # #     plt.xlabel('Horizontal Expansion (%)')
# # # #     plt.ylabel('Vertical Lift (%)')
# # # #     plt.legend()
# # # #     plt.grid(True)
# # # #     plt.axis('equal') # アスペクト比を1:1にして歪みを防ぐ

# # # #     # 角度計算（最大挙上時の角度）
# # # #     angle_l = math.degrees(math.atan2(L_dy[max_idx_l], L_dx[max_idx_l]))
# # # #     angle_r = math.degrees(math.atan2(R_dy[max_idx_r], R_dx[max_idx_r]))
    
# # # #     info_text = f"Left Angle: {angle_l:.1f} deg\nRight Angle: {angle_r:.1f} deg"
# # # #     plt.text(0.05, 0.95, info_text, transform=plt.gca().transAxes, 
# # # #              fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# # # #     plt.savefig(GRAPH_PATH_TRAJECTORY)
# # # #     print(f"軌跡グラフを保存しました: {GRAPH_PATH_TRAJECTORY}")
    
# # # #     # 数値出力
# # # #     print("\n" + "="*40)
# # # #     print("【解析結果】")
# # # #     print(f"Left Max Lift : {np.max(L_dy):.2f}% (Angle: {angle_l:.1f}°)")
# # # #     print(f"Right Max Lift: {np.max(R_dy):.2f}% (Angle: {angle_r:.1f}°)")
# # # #     print("="*40 + "\n")

# # # #     plt.show()

# # # # if __name__ == "__main__":
# # # #     record_and_analyze()

# # # import cv2
# # # import mediapipe as mp
# # # import numpy as np
# # # import matplotlib.pyplot as plt
# # # import time
# # # import os
# # # import math

# # # # ==========================================
# # # # 設定エリア
# # # # ==========================================
# # # # あなたの指定したパスを設定（エラーが出ないよう、先ほどの成功例に合わせています）
# # # RESULT_DIR = '/Users/takahiro/Documents/face_recognition/taki'

# # # # 保存ファイル名（左右分割版として名前を変えています）
# # # VIDEO_PATH = os.path.join(RESULT_DIR, 'split_detail_video.mp4')
# # # GRAPH_PATH_COMPONENTS = os.path.join(RESULT_DIR, 'split_components.png') # 成分分解（時系列）
# # # GRAPH_PATH_TRAJECTORY = os.path.join(RESULT_DIR, 'split_trajectory.png') # 軌跡（地図）

# # # # ディレクトリ生成（念の為）
# # # os.makedirs(RESULT_DIR, exist_ok=True)

# # # # MediaPipe設定
# # # mp_face_mesh = mp.solutions.face_mesh
# # # face_mesh = mp_face_mesh.FaceMesh(
# # #     max_num_faces=1,
# # #     refine_landmarks=True,
# # #     min_detection_confidence=0.5,
# # #     min_tracking_confidence=0.5)

# # # # ランドマークID
# # # ID_EYE_L = 33
# # # ID_EYE_R = 263
# # # ID_NOSE_BRIDGE = 168 
# # # ID_MOUTH_L = 61
# # # ID_MOUTH_R = 291

# # # def get_coords(landmarks, idx, w, h):
# # #     return np.array([landmarks[idx].x * w, landmarks[idx].y * h])

# # # def rotate_point(point, pivot, angle_rad):
# # #     px, py = point
# # #     cx, cy = pivot
# # #     tx, ty = px - cx, py - cy
# # #     cos_a = np.cos(angle_rad)
# # #     sin_a = np.sin(angle_rad)
# # #     nx = tx * cos_a - ty * sin_a
# # #     ny = tx * sin_a + ty * cos_a
# # #     return np.array([nx + cx, ny + cy])

# # # def record_and_analyze():
# # #     cap = cv2.VideoCapture(0)
# # #     if not cap.isOpened():
# # #         print("カメラが見つかりません")
# # #         return

# # #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# # #     fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
# # #     w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# # #     h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# # #     writer = cv2.VideoWriter(VIDEO_PATH, fourcc, fps, (w, h))

# # #     frames_data = [] 
    
# # #     print("--- 左右分割詳細分析モード ---")
# # #     print(" [SPACE]: 録画開始（真顔 -> 笑顔）")
# # #     print(" 左右のグラフを分けて出力します。")

# # #     is_recording = False
# # #     start_time = 0
    
# # #     baseline_l = None
# # #     baseline_r = None
# # #     calibration_frames = []
# # #     CALIBRATION_COUNT = 5

# # #     while True:
# # #         success, image = cap.read()
# # #         if not success: break

# # #         image = cv2.flip(image, 1)
# # #         display_image = image.copy()
# # #         rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# # #         results = face_mesh.process(rgb_image)

# # #         if results.multi_face_landmarks:
# # #             lm = results.multi_face_landmarks[0].landmark
            
# # #             p_eye_l = get_coords(lm, ID_EYE_L, w, h)
# # #             p_eye_r = get_coords(lm, ID_EYE_R, w, h)
# # #             p_nose = get_coords(lm, ID_NOSE_BRIDGE, w, h)
# # #             p_mouth_l = get_coords(lm, ID_MOUTH_L, w, h)
# # #             p_mouth_r = get_coords(lm, ID_MOUTH_R, w, h)

# # #             # 傾き補正
# # #             delta_eyes = p_eye_r - p_eye_l
# # #             face_angle = np.arctan2(delta_eyes[1], delta_eyes[0])
# # #             fixed_mouth_l = rotate_point(p_mouth_l, p_nose, -face_angle)
# # #             fixed_mouth_r = rotate_point(p_mouth_r, p_nose, -face_angle)
            
# # #             face_width = np.linalg.norm(p_eye_r - p_eye_l)

# # #             # 描画
# # #             cv2.line(image, tuple(p_nose.astype(int)), tuple(p_mouth_l.astype(int)), (0, 255, 0), 1)
# # #             cv2.line(image, tuple(p_nose.astype(int)), tuple(p_mouth_r.astype(int)), (0, 255, 0), 1)

# # #             if is_recording:
# # #                 elapsed = time.time() - start_time
                
# # #                 if len(calibration_frames) < CALIBRATION_COUNT:
# # #                     calibration_frames.append({'L': fixed_mouth_l, 'R': fixed_mouth_r})
# # #                     cv2.putText(display_image, "Calibrating...", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                
# # #                 else:
# # #                     if baseline_l is None:
# # #                         l_points = np.array([f['L'] for f in calibration_frames])
# # #                         r_points = np.array([f['R'] for f in calibration_frames])
# # #                         baseline_l = np.mean(l_points, axis=0)
# # #                         baseline_r = np.mean(r_points, axis=0)

# # #                     vec_l = (fixed_mouth_l - baseline_l) / face_width
# # #                     vec_r = (fixed_mouth_r - baseline_r) / face_width
                    
# # #                     l_dx = -vec_l[0] 
# # #                     l_dy = -vec_l[1]
# # #                     r_dx = vec_r[0]
# # #                     r_dy = -vec_r[1]

# # #                     frames_data.append({
# # #                         'time': elapsed,
# # #                         'L_dx': l_dx, 'L_dy': l_dy,
# # #                         'R_dx': r_dx, 'R_dy': r_dy
# # #                     })

# # #                     cv2.putText(display_image, "REC", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# # #                 if elapsed > 4.0:
# # #                     break
            
# # #             writer.write(image)

# # #         cv2.imshow('Split Analysis', display_image)
# # #         key = cv2.waitKey(5)
# # #         if key == 32 and not is_recording:
# # #             is_recording = True
# # #             start_time = time.time()
# # #         elif key == 27:
# # #             break

# # #     cap.release()
# # #     writer.release()
# # #     cv2.destroyAllWindows()

# # #     if frames_data:
# # #         analyze_vectors_split(frames_data)

# # # def analyze_vectors_split(data):
# # #     times = [d['time'] for d in data]
# # #     L_dx = [d['L_dx'] * 100 for d in data]
# # #     L_dy = [d['L_dy'] * 100 for d in data]
# # #     R_dx = [d['R_dx'] * 100 for d in data]
# # #     R_dy = [d['R_dy'] * 100 for d in data]

# # #     # --- グラフ1: 時系列（上下分割：上が左、下が右） ---
# # #     # sharex=Trueで時間軸を共有、sharey=Trueで縦軸のスケールを統一して比較しやすくする
# # #     fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, sharey=True)
    
# # #     # 上段：左側の動き (Left)
# # #     ax1.plot(times, L_dy, label='Vertical Lift (UP)', color='blue', linewidth=2)
# # #     ax1.plot(times, L_dx, label='Horizontal Widen (OUT)', color='cyan', linestyle='--', linewidth=2)
# # #     ax1.set_title('Left Side Movement (Left Mouth Corner)')
# # #     ax1.set_ylabel('Movement (%)')
# # #     ax1.legend(loc='upper left')
# # #     ax1.grid(True)
# # #     ax1.axhline(0, color='black', linewidth=0.5)

# # #     # 下段：右側の動き (Right)
# # #     ax2.plot(times, R_dy, label='Vertical Lift (UP)', color='orange', linewidth=2)
# # #     ax2.plot(times, R_dx, label='Horizontal Widen (OUT)', color='red', linestyle='--', linewidth=2)
# # #     ax2.set_title('Right Side Movement (Right Mouth Corner)')
# # #     ax2.set_xlabel('Time (sec)')
# # #     ax2.set_ylabel('Movement (%)')
# # #     ax2.legend(loc='upper left')
# # #     ax2.grid(True)
# # #     ax2.axhline(0, color='black', linewidth=0.5)
    
# # #     plt.tight_layout()
# # #     plt.savefig(GRAPH_PATH_COMPONENTS)
# # #     print(f"時系列グラフを保存しました: {GRAPH_PATH_COMPONENTS}")

# # #     # --- グラフ2: 軌跡（左右分割：左枠が左、右枠が右） ---
# # #     fig2, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 6))
    
# # #     # スケールを統一するための計算（左右で最大の方に合わせる）
# # #     all_vals = L_dx + L_dy + R_dx + R_dy
# # #     max_val = max(np.max(np.abs(all_vals)), 5) # 最低でも5%の幅は持たせる
# # #     limit = max_val * 1.2 # 余白を持たせる
    
# # #     # 左側の軌跡
# # #     ax_l.plot(L_dx, L_dy, color='blue', marker='o', markersize=3, alpha=0.5)
# # #     ax_l.plot(L_dx[0], L_dy[0], 'bx', label='Start', markersize=8)
# # #     ax_l.plot(L_dx[np.argmax(L_dy)], np.max(L_dy), 'bo', fillstyle='none', markersize=10, label='Max Lift')
# # #     ax_l.set_title('Left Trajectory')
# # #     ax_l.set_xlabel('Horizontal Widen (%)')
# # #     ax_l.set_ylabel('Vertical Lift (%)')
# # #     ax_l.grid(True)
# # #     ax_l.axhline(0, color='black', linewidth=0.5)
# # #     ax_l.axvline(0, color='black', linewidth=0.5)
# # #     ax_l.set_xlim(-limit/2, limit) # 横方向は外側への動きがメインなので、マイナス側は少なめに
# # #     ax_l.set_ylim(-limit/2, limit)
# # #     ax_l.set_aspect('equal') # 正方形にする
    
# # #     # 角度計算
# # #     angle_l = math.degrees(math.atan2(np.max(L_dy), L_dx[np.argmax(L_dy)]))
# # #     ax_l.text(0.05, 0.95, f"Angle: {angle_l:.1f}°", transform=ax_l.transAxes, 
# # #               bbox=dict(facecolor='white', alpha=0.8))

# # #     # 右側の軌跡
# # #     ax_r.plot(R_dx, R_dy, color='orange', marker='o', markersize=3, alpha=0.5)
# # #     ax_r.plot(R_dx[0], R_dy[0], 'rx', label='Start', markersize=8)
# # #     ax_r.plot(R_dx[np.argmax(R_dy)], np.max(R_dy), 'ro', fillstyle='none', markersize=10, label='Max Lift')
# # #     ax_r.set_title('Right Trajectory')
# # #     ax_r.set_xlabel('Horizontal Widen (%)')
# # #     # Y軸ラベルは左と同じなので省略しても良いが、あっても良い
# # #     ax_r.grid(True)
# # #     ax_r.axhline(0, color='black', linewidth=0.5)
# # #     ax_r.axvline(0, color='black', linewidth=0.5)
# # #     ax_r.set_xlim(-limit/2, limit)
# # #     ax_r.set_ylim(-limit/2, limit)
# # #     ax_r.set_aspect('equal')

# # #     angle_r = math.degrees(math.atan2(np.max(R_dy), R_dx[np.argmax(R_dy)]))
# # #     ax_r.text(0.05, 0.95, f"Angle: {angle_r:.1f}°", transform=ax_r.transAxes, 
# # #               bbox=dict(facecolor='white', alpha=0.8))

# # #     plt.tight_layout()
# # #     plt.savefig(GRAPH_PATH_TRAJECTORY)
# # #     print(f"軌跡グラフを保存しました: {GRAPH_PATH_TRAJECTORY}")
    
# # #     plt.show()

# # # if __name__ == "__main__":
# # #     record_and_analyze()

# # import cv2
# # import mediapipe as mp
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import time
# # import os
# # import math

# # # ==========================================
# # # 設定エリア（物理量の基準）
# # # ==========================================
# # # 保存先（あなたの環境に合わせています）
# # RESULT_DIR = '/Users/takahiro/Documents/face_recognition/taki'

# # # 【重要】基準となる実測値
# # # 一般的な成人の目尻〜目尻の幅はおよそ95〜105mmです。
# # # 自分の顔で定規で測ってここを書き換えると、より正確になります。
# # REAL_EYE_WIDTH_MM = 105.0 

# # # 保存ファイル名
# # VIDEO_PATH = os.path.join(RESULT_DIR, 'smile_mm_video.mp4')
# # GRAPH_PATH_COMPONENTS = os.path.join(RESULT_DIR, 'smile_mm_components.png')
# # GRAPH_PATH_TRAJECTORY = os.path.join(RESULT_DIR, 'smile_mm_trajectory.png')

# # os.makedirs(RESULT_DIR, exist_ok=True)

# # # MediaPipe設定
# # mp_face_mesh = mp.solutions.face_mesh
# # face_mesh = mp_face_mesh.FaceMesh(
# #     max_num_faces=1,
# #     refine_landmarks=True,
# #     min_detection_confidence=0.5,
# #     min_tracking_confidence=0.5)

# # # ランドマークID
# # ID_EYE_L_OUTER = 33
# # ID_EYE_R_OUTER = 263
# # ID_NOSE_BRIDGE = 168 
# # ID_MOUTH_L = 61
# # ID_MOUTH_R = 291

# # def get_coords(landmarks, idx, w, h):
# #     return np.array([landmarks[idx].x * w, landmarks[idx].y * h])

# # def rotate_point(point, pivot, angle_rad):
# #     """顔の傾きを補正して、首を傾げても数値がズレないようにする"""
# #     px, py = point
# #     cx, cy = pivot
# #     tx, ty = px - cx, py - cy
# #     cos_a = np.cos(angle_rad)
# #     sin_a = np.sin(angle_rad)
# #     nx = tx * cos_a - ty * sin_a
# #     ny = tx * sin_a + ty * cos_a
# #     return np.array([nx + cx, ny + cy])

# # def record_and_analyze_mm():
# #     cap = cv2.VideoCapture(0)
# #     if not cap.isOpened():
# #         print("カメラが見つかりません")
# #         return

# #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# #     fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
# #     w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# #     h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# #     writer = cv2.VideoWriter(VIDEO_PATH, fourcc, fps, (w, h))

# #     frames_data = [] 
    
# #     print(f"--- ミリメートル(mm)計測モード ---")
# #     print(f"基準: 目尻間の幅 = {REAL_EYE_WIDTH_MM}mm")
# #     print(" [SPACE]: 録画開始（真顔 -> 笑顔）")

# #     is_recording = False
# #     start_time = 0
    
# #     baseline_l = None
# #     baseline_r = None
# #     calibration_frames = []
# #     CALIBRATION_COUNT = 5

# #     while True:
# #         success, image = cap.read()
# #         if not success: break

# #         image = cv2.flip(image, 1)
# #         display_image = image.copy()
# #         rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# #         results = face_mesh.process(rgb_image)

# #         if results.multi_face_landmarks:
# #             lm = results.multi_face_landmarks[0].landmark
            
# #             # 座標取得
# #             p_eye_l = get_coords(lm, ID_EYE_L_OUTER, w, h)
# #             p_eye_r = get_coords(lm, ID_EYE_R_OUTER, w, h)
# #             p_nose = get_coords(lm, ID_NOSE_BRIDGE, w, h)
# #             p_mouth_l = get_coords(lm, ID_MOUTH_L, w, h)
# #             p_mouth_r = get_coords(lm, ID_MOUTH_R, w, h)

# #             # 1. スケール計算 (mm/px) - 毎フレーム計算することで前後移動に対応
# #             eye_width_px = np.linalg.norm(p_eye_r - p_eye_l)
# #             if eye_width_px == 0: eye_width_px = 1 # ゼロ除算防止
# #             mm_per_px = REAL_EYE_WIDTH_MM / eye_width_px

# #             # 2. 傾き補正
# #             delta_eyes = p_eye_r - p_eye_l
# #             face_angle = np.arctan2(delta_eyes[1], delta_eyes[0])
            
# #             # 鼻を中心に回転させて水平にする
# #             fixed_mouth_l = rotate_point(p_mouth_l, p_nose, -face_angle)
# #             fixed_mouth_r = rotate_point(p_mouth_r, p_nose, -face_angle)
            
# #             # 描画（現在値をmmで表示）
# #             cv2.line(display_image, tuple(p_eye_l.astype(int)), tuple(p_eye_r.astype(int)), (255, 255, 0), 1)
# #             cv2.putText(display_image, f"Scale: {mm_per_px:.2f} mm/px", (20, 80), 
# #                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

# #             if is_recording:
# #                 elapsed = time.time() - start_time
                
# #                 # 最初の数フレームで「真顔の位置」を平均化して基準にする
# #                 if len(calibration_frames) < CALIBRATION_COUNT:
# #                     calibration_frames.append({'L': fixed_mouth_l, 'R': fixed_mouth_r})
# #                     cv2.putText(display_image, "Calibrating Neutral...", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                
# #                 else:
# #                     if baseline_l is None:
# #                         l_points = np.array([f['L'] for f in calibration_frames])
# #                         r_points = np.array([f['R'] for f in calibration_frames])
# #                         baseline_l = np.mean(l_points, axis=0)
# #                         baseline_r = np.mean(r_points, axis=0)

# #                     # 3. 移動量計算 (ピクセル)
# #                     # 左口角: 左へ行くとマイナスなので、外側(左)への動きを正にするために反転
# #                     # Y軸: 下に行くとプラスなので、上への動きを正にするために反転
# #                     vec_l_px = fixed_mouth_l - baseline_l
# #                     vec_r_px = fixed_mouth_r - baseline_r
                    
# #                     # 4. mmに変換
# #                     l_dx_mm = -vec_l_px[0] * mm_per_px # 外側への広がり
# #                     l_dy_mm = -vec_l_px[1] * mm_per_px # 上への引き上げ
                    
# #                     r_dx_mm = vec_r_px[0] * mm_per_px  # 外側への広がり
# #                     r_dy_mm = -vec_r_px[1] * mm_per_px # 上への引き上げ

# #                     frames_data.append({
# #                         'time': elapsed,
# #                         'L_dx': l_dx_mm, 'L_dy': l_dy_mm,
# #                         'R_dx': r_dx_mm, 'R_dy': r_dy_mm
# #                     })

# #                     cv2.putText(display_image, "REC", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
# #                     # リアルタイム数値表示
# #                     info_l = f"L: Up={l_dy_mm:.1f}mm Out={l_dx_mm:.1f}mm"
# #                     info_r = f"R: Up={r_dy_mm:.1f}mm Out={r_dx_mm:.1f}mm"
# #                     cv2.putText(display_image, info_l, (20, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 200), 2)
# #                     cv2.putText(display_image, info_r, (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)

# #                 if elapsed > 4.0:
# #                     break
            
# #             writer.write(display_image)

# #         cv2.imshow('MM Analysis', display_image)
# #         key = cv2.waitKey(5)
# #         if key == 32 and not is_recording: # Space
# #             is_recording = True
# #             start_time = time.time()
# #         elif key == 27: # Esc
# #             break

# #     cap.release()
# #     writer.release()
# #     cv2.destroyAllWindows()

# #     if frames_data:
# #         analyze_vectors_mm(frames_data)

# # def analyze_vectors_mm(data):
# #     times = [d['time'] for d in data]
# #     L_dx = [d['L_dx'] for d in data]
# #     L_dy = [d['L_dy'] for d in data]
# #     R_dx = [d['R_dx'] for d in data]
# #     R_dy = [d['R_dy'] for d in data]

# #     # --- グラフ1: 時系列（mm単位） ---
# #     fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, sharey=True)
    
# #     # 共通設定：Y軸の範囲を固定（例：-5mm 〜 20mm）すると比較しやすい
# #     # ここでは自動調整に任せますが、データによっては set_ylim 推奨
    
# #     # Left
# #     ax1.plot(times, L_dy, label='Vertical Lift (UP)', color='blue', linewidth=2)
# #     ax1.plot(times, L_dx, label='Horizontal Widen (OUT)', color='cyan', linestyle='--', linewidth=2)
# #     ax1.set_title('Left Side Movement [mm]')
# #     ax1.set_ylabel('Distance (mm)')
# #     ax1.legend(loc='upper left')
# #     ax1.grid(True)
# #     ax1.axhline(0, color='black', linewidth=0.5)

# #     # Right
# #     ax2.plot(times, R_dy, label='Vertical Lift (UP)', color='orange', linewidth=2)
# #     ax2.plot(times, R_dx, label='Horizontal Widen (OUT)', color='red', linestyle='--', linewidth=2)
# #     ax2.set_title('Right Side Movement [mm]')
# #     ax2.set_xlabel('Time (sec)')
# #     ax2.set_ylabel('Distance (mm)')
# #     ax2.legend(loc='upper left')
# #     ax2.grid(True)
# #     ax2.axhline(0, color='black', linewidth=0.5)
    
# #     plt.tight_layout()
# #     plt.savefig(GRAPH_PATH_COMPONENTS)

# #     # --- グラフ2: 軌跡（mm単位） ---
# #     fig2, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 6))
    
# #     # スケール統一
# #     all_vals = L_dx + L_dy + R_dx + R_dy
# #     max_val = max(np.max(np.abs(all_vals)), 10) # 最低でも10mm幅
# #     limit = max_val * 1.2
    
# #     # Left Trajectory
# #     ax_l.plot(L_dx, L_dy, color='blue', marker='o', markersize=3, alpha=0.5)
# #     ax_l.plot(L_dx[0], L_dy[0], 'bx', label='Start', markersize=8)
# #     ax_l.plot(L_dx[np.argmax(L_dy)], np.max(L_dy), 'bo', fillstyle='none', markersize=10, label='Max Lift')
# #     ax_l.set_title('Left Trajectory (mm)')
# #     ax_l.set_xlabel('Horizontal Widen (mm)')
# #     ax_l.set_ylabel('Vertical Lift (mm)')
# #     ax_l.grid(True)
# #     ax_l.axhline(0, color='black', linewidth=0.5)
# #     ax_l.axvline(0, color='black', linewidth=0.5)
# #     ax_l.set_xlim(-5, limit) 
# #     ax_l.set_ylim(-5, limit)
# #     ax_l.set_aspect('equal')
    
# #     angle_l = math.degrees(math.atan2(np.max(L_dy), L_dx[np.argmax(L_dy)]))
# #     ax_l.text(0.05, 0.95, f"Angle: {angle_l:.1f}°", transform=ax_l.transAxes, bbox=dict(facecolor='white', alpha=0.8))

# #     # Right Trajectory
# #     ax_r.plot(R_dx, R_dy, color='orange', marker='o', markersize=3, alpha=0.5)
# #     ax_r.plot(R_dx[0], R_dy[0], 'rx', label='Start', markersize=8)
# #     ax_r.plot(R_dx[np.argmax(R_dy)], np.max(R_dy), 'ro', fillstyle='none', markersize=10, label='Max Lift')
# #     ax_r.set_title('Right Trajectory (mm)')
# #     ax_r.set_xlabel('Horizontal Widen (mm)')
# #     ax_r.grid(True)
# #     ax_r.axhline(0, color='black', linewidth=0.5)
# #     ax_r.axvline(0, color='black', linewidth=0.5)
# #     ax_r.set_xlim(-5, limit)
# #     ax_r.set_ylim(-5, limit)
# #     ax_r.set_aspect('equal')

# #     angle_r = math.degrees(math.atan2(np.max(R_dy), R_dx[np.argmax(R_dy)]))
# #     ax_r.text(0.05, 0.95, f"Angle: {angle_r:.1f}°", transform=ax_r.transAxes, bbox=dict(facecolor='white', alpha=0.8))

# #     plt.tight_layout()
# #     plt.savefig(GRAPH_PATH_TRAJECTORY)
# #     print(f"グラフを保存しました:\n {GRAPH_PATH_COMPONENTS}\n {GRAPH_PATH_TRAJECTORY}")
    
# #     plt.show()

# # if __name__ == "__main__":
# #     record_and_analyze_mm()

# import cv2
# import mediapipe as mp
# import numpy as np
# import matplotlib.pyplot as plt
# import time
# import os
# import math

# # ==========================================
# # 設定エリア
# # ==========================================
# RESULT_DIR = '/Users/takahiro/Documents/face_recognition/taki'
# REAL_EYE_WIDTH_MM = 105.0  # 基準となる目尻間の幅(mm)

# # 保存ファイル名
# VIDEO_PATH = os.path.join(RESULT_DIR, 'smile_mm_video_fixed.mp4')
# GRAPH_PATH_COMPONENTS = os.path.join(RESULT_DIR, 'smile_mm_components.png')
# GRAPH_PATH_TRAJECTORY = os.path.join(RESULT_DIR, 'smile_mm_trajectory.png')

# os.makedirs(RESULT_DIR, exist_ok=True)

# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(
#     max_num_faces=1,
#     refine_landmarks=True,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5)

# # ランドマークID
# ID_EYE_L_OUTER = 33
# ID_EYE_R_OUTER = 263
# ID_NOSE_BRIDGE = 168 
# ID_MOUTH_L = 61
# ID_MOUTH_R = 291

# def get_coords(landmarks, idx, w, h):
#     return np.array([landmarks[idx].x * w, landmarks[idx].y * h])

# def rotate_point(point, pivot, angle_rad):
#     px, py = point
#     cx, cy = pivot
#     tx, ty = px - cx, py - cy
#     cos_a = np.cos(angle_rad)
#     sin_a = np.sin(angle_rad)
#     nx = tx * cos_a - ty * sin_a
#     ny = tx * sin_a + ty * cos_a
#     return np.array([nx + cx, ny + cy])

# def record_and_analyze_mm_fixed():
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("カメラが見つかりません")
#         return

#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
#     w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     writer = cv2.VideoWriter(VIDEO_PATH, fourcc, fps, (w, h))

#     frames_data = [] 
    
#     print(f"--- 口角追跡・mm計測モード (修正版) ---")
#     print(f"基準: 目尻間の幅 = {REAL_EYE_WIDTH_MM}mm")
#     print(" [SPACE]: 録画開始")

#     is_recording = False
#     start_time = 0
    
#     baseline_l = None
#     baseline_r = None
#     calibration_frames = []
#     CALIBRATION_COUNT = 5

#     while True:
#         success, image = cap.read()
#         if not success: break

#         image = cv2.flip(image, 1)
#         display_image = image.copy()
#         rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         results = face_mesh.process(rgb_image)

#         if results.multi_face_landmarks:
#             lm = results.multi_face_landmarks[0].landmark
            
#             p_eye_l = get_coords(lm, ID_EYE_L_OUTER, w, h)
#             p_eye_r = get_coords(lm, ID_EYE_R_OUTER, w, h)
#             p_nose = get_coords(lm, ID_NOSE_BRIDGE, w, h)
#             p_mouth_l = get_coords(lm, ID_MOUTH_L, w, h)
#             p_mouth_r = get_coords(lm, ID_MOUTH_R, w, h)

#             # --- 1. スケール計算 (目は定規として使用) ---
#             eye_width_px = np.linalg.norm(p_eye_r - p_eye_l)
#             if eye_width_px == 0: eye_width_px = 1
#             mm_per_px = REAL_EYE_WIDTH_MM / eye_width_px

#             # 目のライン（グレーで薄く表示：定規であることを示す）
#             cv2.line(display_image, tuple(p_eye_l.astype(int)), tuple(p_eye_r.astype(int)), (100, 100, 100), 1)
#             cv2.putText(display_image, "Ruler (Scale Ref)", (int(p_eye_l[0]), int(p_eye_l[1]-10)), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

#             # --- 2. 口角の追跡と描画 (ここがメイン) ---
#             # 口角に緑の点を描画
#             cv2.circle(display_image, tuple(p_mouth_l.astype(int)), 4, (0, 255, 0), -1)
#             cv2.circle(display_image, tuple(p_mouth_r.astype(int)), 4, (0, 255, 0), -1)

#             # 傾き補正計算
#             delta_eyes = p_eye_r - p_eye_l
#             face_angle = np.arctan2(delta_eyes[1], delta_eyes[0])
#             fixed_mouth_l = rotate_point(p_mouth_l, p_nose, -face_angle)
#             fixed_mouth_r = rotate_point(p_mouth_r, p_nose, -face_angle)

#             if is_recording:
#                 elapsed = time.time() - start_time
                
#                 # キャリブレーション（真顔の位置を記録）
#                 if len(calibration_frames) < CALIBRATION_COUNT:
#                     calibration_frames.append({'L': fixed_mouth_l, 'R': fixed_mouth_r})
#                     cv2.putText(display_image, "Calibrating Neutral...", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                
#                 else:
#                     if baseline_l is None:
#                         # 基準点（真顔の口角位置）を決定
#                         baseline_l = np.mean([f['L'] for f in calibration_frames], axis=0)
#                         baseline_r = np.mean([f['R'] for f in calibration_frames], axis=0)

#                     # 移動量計算
#                     vec_l_px = fixed_mouth_l - baseline_l
#                     vec_r_px = fixed_mouth_r - baseline_r
                    
#                     # mm変換
#                     l_dx_mm = -vec_l_px[0] * mm_per_px 
#                     l_dy_mm = -vec_l_px[1] * mm_per_px
#                     r_dx_mm = vec_r_px[0] * mm_per_px  
#                     r_dy_mm = -vec_r_px[1] * mm_per_px 

#                     frames_data.append({
#                         'time': elapsed,
#                         'L_dx': l_dx_mm, 'L_dy': l_dy_mm,
#                         'R_dx': r_dx_mm, 'R_dy': r_dy_mm
#                     })

#                     # --- ベクトル（動きの矢印）を描画 ---
#                     # 画面上で「どれくらい動いたか」を可視化する（視覚的フィードバック）
#                     # 左口角の軌跡
#                     start_l = rotate_point(baseline_l, p_nose, face_angle) # 画面上の座標に戻す
#                     cv2.arrowedLine(display_image, tuple(start_l.astype(int)), tuple(p_mouth_l.astype(int)), (255, 0, 0), 2)
                    
#                     # 右口角の軌跡
#                     start_r = rotate_point(baseline_r, p_nose, face_angle)
#                     cv2.arrowedLine(display_image, tuple(start_r.astype(int)), tuple(p_mouth_r.astype(int)), (0, 0, 255), 2)

#                     cv2.putText(display_image, "REC", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
#                     # 数値表示
#                     cv2.putText(display_image, f"L Move: {np.linalg.norm([l_dx_mm, l_dy_mm]):.1f}mm", 
#                                 (int(p_mouth_l[0]-60), int(p_mouth_l[1]+30)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
#                     cv2.putText(display_image, f"R Move: {np.linalg.norm([r_dx_mm, r_dy_mm]):.1f}mm", 
#                                 (int(p_mouth_r[0]-20), int(p_mouth_r[1]+30)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

#                 if elapsed > 4.0:
#                     break
            
#             writer.write(display_image)

#         cv2.imshow('Mouth Corner Analysis', display_image)
#         key = cv2.waitKey(5)
#         if key == 32 and not is_recording: 
#             is_recording = True
#             start_time = time.time()
#         elif key == 27: 
#             break

#     cap.release()
#     writer.release()
#     cv2.destroyAllWindows()

#     if frames_data:
#         analyze_vectors_mm_fixed(frames_data)

# def analyze_vectors_mm_fixed(data):
#     # グラフ描画部分は前回と同じロジックですが、見やすさのため再記述
#     times = [d['time'] for d in data]
#     L_dx = [d['L_dx'] for d in data]
#     L_dy = [d['L_dy'] for d in data]
#     R_dx = [d['R_dx'] for d in data]
#     R_dy = [d['R_dy'] for d in data]

#     fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, sharey=True)
    
#     # Left
#     ax1.plot(times, L_dy, label='Vertical Lift (UP)', color='blue', linewidth=2)
#     ax1.plot(times, L_dx, label='Horizontal Widen (OUT)', color='cyan', linestyle='--', linewidth=2)
#     ax1.set_title('Left Side Movement [mm]')
#     ax1.set_ylabel('Distance (mm)')
#     ax1.legend(loc='upper left')
#     ax1.grid(True)
#     ax1.axhline(0, color='black', linewidth=0.5)

#     # Right
#     ax2.plot(times, R_dy, label='Vertical Lift (UP)', color='orange', linewidth=2)
#     ax2.plot(times, R_dx, label='Horizontal Widen (OUT)', color='red', linestyle='--', linewidth=2)
#     ax2.set_title('Right Side Movement [mm]')
#     ax2.set_xlabel('Time (sec)')
#     ax2.set_ylabel('Distance (mm)')
#     ax2.legend(loc='upper left')
#     ax2.grid(True)
#     ax2.axhline(0, color='black', linewidth=0.5)
    
#     plt.tight_layout()
#     plt.savefig(GRAPH_PATH_COMPONENTS)

#     # Trajectory
#     fig2, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 6))
    
#     all_vals = L_dx + L_dy + R_dx + R_dy
#     max_val = max(np.max(np.abs(all_vals)), 10)
#     limit = max_val * 1.2
    
#     # Left
#     ax_l.plot(L_dx, L_dy, color='blue', marker='o', markersize=3, alpha=0.5)
#     ax_l.plot(L_dx[0], L_dy[0], 'bx', label='Start', markersize=8)
#     ax_l.plot(L_dx[np.argmax(L_dy)], np.max(L_dy), 'bo', fillstyle='none', markersize=10)
#     ax_l.set_title('Left Trajectory (mm)')
#     ax_l.set_xlabel('Horizontal Widen (mm)')
#     ax_l.set_ylabel('Vertical Lift (mm)')
#     ax_l.grid(True)
#     ax_l.set_xlim(-5, limit) 
#     ax_l.set_ylim(-5, limit)
#     ax_l.set_aspect('equal')
    
#     angle_l = math.degrees(math.atan2(np.max(L_dy), L_dx[np.argmax(L_dy)]))
#     ax_l.text(0.05, 0.95, f"Angle: {angle_l:.1f}°", transform=ax_l.transAxes, bbox=dict(facecolor='white', alpha=0.8))

#     # Right
#     ax_r.plot(R_dx, R_dy, color='orange', marker='o', markersize=3, alpha=0.5)
#     ax_r.plot(R_dx[0], R_dy[0], 'rx', label='Start', markersize=8)
#     ax_r.plot(R_dx[np.argmax(R_dy)], np.max(R_dy), 'ro', fillstyle='none', markersize=10)
#     ax_r.set_title('Right Trajectory (mm)')
#     ax_r.set_xlabel('Horizontal Widen (mm)')
#     ax_r.grid(True)
#     ax_r.set_xlim(-5, limit)
#     ax_r.set_ylim(-5, limit)
#     ax_r.set_aspect('equal')

#     angle_r = math.degrees(math.atan2(np.max(R_dy), R_dx[np.argmax(R_dy)]))
#     ax_r.text(0.05, 0.95, f"Angle: {angle_r:.1f}°", transform=ax_r.transAxes, bbox=dict(facecolor='white', alpha=0.8))

#     plt.tight_layout()
#     plt.savefig(GRAPH_PATH_TRAJECTORY)
    
#     plt.show()

# if __name__ == "__main__":
#     record_and_analyze_mm_fixed()

import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import math

# ==========================================
# 設定エリア
# ==========================================
# 保存先ディレクトリ（必要に応じて変更してください）
RESULT_DIR = './webcam_results_mm' 
os.makedirs(RESULT_DIR, exist_ok=True)

# 【重要】基準となる実測値
# 一般的な成人の目尻〜目尻の幅はおよそ95〜105mmです。
REAL_EYE_WIDTH_MM = 105.0 

# 保存ファイル名
VIDEO_PATH = os.path.join(RESULT_DIR, 'smile_mm_analysis.mp4')
GRAPH_PATH_COMPONENTS = os.path.join(RESULT_DIR, 'smile_mm_components.png')
GRAPH_PATH_TRAJECTORY = os.path.join(RESULT_DIR, 'smile_mm_trajectory.png')

# MediaPipe設定
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# ランドマークID
ID_EYE_L_OUTER = 33
ID_EYE_R_OUTER = 263
ID_NOSE_BRIDGE = 168 
ID_MOUTH_L = 61
ID_MOUTH_R = 291

def get_coords(landmarks, idx, w, h):
    return np.array([landmarks[idx].x * w, landmarks[idx].y * h])

def rotate_point(point, pivot, angle_rad):
    """顔の傾き補正"""
    px, py = point
    cx, cy = pivot
    tx, ty = px - cx, py - cy
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    nx = tx * cos_a - ty * sin_a
    ny = tx * sin_a + ty * cos_a
    return np.array([nx + cx, ny + cy])

def record_and_analyze_mm_fixed():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("カメラが見つかりません")
        return

    # MacBookなど高解像度対応
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(VIDEO_PATH, fourcc, fps, (w, h))

    frames_data = [] 
    
    print(f"--- 口角追跡・mm計測モード (修正版) ---")
    print(f"基準: 目尻間の幅 = {REAL_EYE_WIDTH_MM}mm")
    print(" [SPACE]: 録画開始")
    print(" [ESC]: 終了")

    is_recording = False
    start_time = 0
    
    baseline_l = None
    baseline_r = None
    calibration_frames = []
    CALIBRATION_COUNT = 10 # キャリブレーションフレーム数を少し増やす

    while True:
        success, image = cap.read()
        if not success: break

        image = cv2.flip(image, 1)
        display_image = image.copy()
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            
            p_eye_l = get_coords(lm, ID_EYE_L_OUTER, w, h)
            p_eye_r = get_coords(lm, ID_EYE_R_OUTER, w, h)
            p_nose = get_coords(lm, ID_NOSE_BRIDGE, w, h)
            p_mouth_l = get_coords(lm, ID_MOUTH_L, w, h)
            p_mouth_r = get_coords(lm, ID_MOUTH_R, w, h)

            # --- 1. スケール計算 (mm/px) ---
            eye_width_px = np.linalg.norm(p_eye_r - p_eye_l)
            if eye_width_px == 0: eye_width_px = 1
            mm_per_px = REAL_EYE_WIDTH_MM / eye_width_px

            # --- 2. 視覚的ガイド (目元のラインは削除し、口角のみ強調) ---
            # 口角に緑のマーカーを表示
            cv2.circle(display_image, tuple(p_mouth_l.astype(int)), 5, (0, 255, 0), -1) # 左口角
            cv2.circle(display_image, tuple(p_mouth_r.astype(int)), 5, (0, 255, 0), -1) # 右口角

            # 傾き補正用計算
            delta_eyes = p_eye_r - p_eye_l
            face_angle = np.arctan2(delta_eyes[1], delta_eyes[0])
            
            # 補正後の座標
            fixed_mouth_l = rotate_point(p_mouth_l, p_nose, -face_angle)
            fixed_mouth_r = rotate_point(p_mouth_r, p_nose, -face_angle)

            if is_recording:
                elapsed = time.time() - start_time
                
                # --- キャリブレーション（真顔の基準位置を取得）---
                if len(calibration_frames) < CALIBRATION_COUNT:
                    calibration_frames.append({'L': fixed_mouth_l, 'R': fixed_mouth_r})
                    cv2.putText(display_image, "Stay Neutral...", (50, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
                
                else:
                    if baseline_l is None:
                        baseline_l = np.mean([f['L'] for f in calibration_frames], axis=0)
                        baseline_r = np.mean([f['R'] for f in calibration_frames], axis=0)
                        print("基準点設定完了。計測中...")

                    # --- 移動量計算 ---
                    vec_l_px = fixed_mouth_l - baseline_l
                    vec_r_px = fixed_mouth_r - baseline_r
                    
                    # mm変換 (左: 外側=マイナス方向を反転, 上=マイナス方向を反転)
                    l_dx_mm = -vec_l_px[0] * mm_per_px 
                    l_dy_mm = -vec_l_px[1] * mm_per_px
                    
                    # mm変換 (右: 外側=プラス方向, 上=マイナス方向を反転)
                    r_dx_mm = vec_r_px[0] * mm_per_px  
                    r_dy_mm = -vec_r_px[1] * mm_per_px 

                    frames_data.append({
                        'time': elapsed,
                        'L_dx': l_dx_mm, 'L_dy': l_dy_mm,
                        'R_dx': r_dx_mm, 'R_dy': r_dy_mm
                    })

                    # --- 【新機能】動きのベクトル（矢印）を表示 ---
                    # 実際の画面上でのベクトル開始位置（真顔位置）を計算
                    start_l_disp = rotate_point(baseline_l, p_nose, face_angle)
                    start_r_disp = rotate_point(baseline_r, p_nose, face_angle)
                    
                    # 矢印を描画 (スケールを少し強調して見やすくする: x2)
                    scale_vis = 1.0
                    end_l_disp = tuple((p_mouth_l).astype(int))
                    end_r_disp = tuple((p_mouth_r).astype(int))

                    # 左矢印 (青)
                    cv2.arrowedLine(display_image, tuple(start_l_disp.astype(int)), end_l_disp, (255, 200, 0), 4)
                    # 右矢印 (赤)
                    cv2.arrowedLine(display_image, tuple(start_r_disp.astype(int)), end_r_disp, (0, 0, 255), 4)

                    cv2.putText(display_image, "RECORDING", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    # 数値表示 (リアルタイム)
                    info_l = f"L: Up={l_dy_mm:.1f}mm / Out={l_dx_mm:.1f}mm"
                    info_r = f"R: Up={r_dy_mm:.1f}mm / Out={r_dx_mm:.1f}mm"
                    cv2.putText(display_image, info_l, (30, h - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
                    cv2.putText(display_image, info_r, (30, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 255), 2)

                if elapsed > 4.5: # 4.5秒で自動停止
                    break
            
            else:
                cv2.putText(display_image, "Press SPACE to Record", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            writer.write(display_image)

        cv2.imshow('Mouth Corner Analysis (Fixed)', display_image)
        key = cv2.waitKey(5)
        if key == 32 and not is_recording: 
            is_recording = True
            start_time = time.time()
        elif key == 27: 
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    if frames_data:
        analyze_vectors_mm_fixed(frames_data)

def analyze_vectors_mm_fixed(data):
    times = [d['time'] for d in data]
    L_dx = [d['L_dx'] for d in data]
    L_dy = [d['L_dy'] for d in data]
    R_dx = [d['R_dx'] for d in data]
    R_dy = [d['R_dy'] for d in data]

    # --- グラフ描画 ---
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, sharey=True)
    
    # Left Graph
    ax1.plot(times, L_dy, label='Vertical Lift (UP)', color='blue', linewidth=2.5)
    ax1.plot(times, L_dx, label='Horizontal Widen (OUT)', color='cyan', linestyle='--', linewidth=2)
    ax1.set_title('Left Side Movement [mm] - (Ideal Pattern)')
    ax1.set_ylabel('Distance (mm)')
    ax1.legend(loc='upper left')
    ax1.grid(True)
    ax1.axhline(0, color='black', linewidth=0.5)

    # Right Graph
    ax2.plot(times, R_dy, label='Vertical Lift (UP)', color='orange', linewidth=2.5)
    ax2.plot(times, R_dx, label='Horizontal Widen (OUT)', color='red', linestyle='--', linewidth=2)
    ax2.set_title('Right Side Movement [mm]')
    ax2.set_xlabel('Time (sec)')
    ax2.set_ylabel('Distance (mm)')
    ax2.legend(loc='upper left')
    ax2.grid(True)
    ax2.axhline(0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(GRAPH_PATH_COMPONENTS)

    # Trajectory Graph
    fig2, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 6))
    
    # 軸のスケールを統一して比較しやすくする
    limit = max(np.max(np.abs(L_dx + L_dy + R_dx + R_dy)), 15) * 1.1
    
    # Left Trajectory
    ax_l.plot(L_dx, L_dy, color='blue', marker='o', markersize=3, alpha=0.5)
    ax_l.plot(L_dx[0], L_dy[0], 'bx', label='Start', markersize=8)
    ax_l.plot(L_dx[np.argmax(L_dy)], np.max(L_dy), 'bo', fillstyle='none', markersize=10)
    ax_l.set_title('Left Trajectory (mm)')
    ax_l.set_xlabel('Horizontal Widen (mm)')
    ax_l.set_ylabel('Vertical Lift (mm)')
    ax_l.grid(True)
    ax_l.set_xlim(-5, limit) 
    ax_l.set_ylim(-5, limit)
    ax_l.set_aspect('equal')
    
    # 角度計算
    idx_max_l = np.argmax(L_dy)
    if L_dx[idx_max_l] != 0:
        angle_l = math.degrees(math.atan2(L_dy[idx_max_l], L_dx[idx_max_l]))
    else: angle_l = 90
    ax_l.text(0.05, 0.95, f"Angle: {angle_l:.1f}°", transform=ax_l.transAxes, bbox=dict(facecolor='white', alpha=0.8))

    # Right Trajectory
    ax_r.plot(R_dx, R_dy, color='orange', marker='o', markersize=3, alpha=0.5)
    ax_r.plot(R_dx[0], R_dy[0], 'rx', label='Start', markersize=8)
    ax_r.plot(R_dx[np.argmax(R_dy)], np.max(R_dy), 'ro', fillstyle='none', markersize=10)
    ax_r.set_title('Right Trajectory (mm)')
    ax_r.set_xlabel('Horizontal Widen (mm)')
    ax_r.grid(True)
    ax_r.set_xlim(-5, limit)
    ax_r.set_ylim(-5, limit)
    ax_r.set_aspect('equal')

    idx_max_r = np.argmax(R_dy)
    if R_dx[idx_max_r] != 0:
        angle_r = math.degrees(math.atan2(R_dy[idx_max_r], R_dx[idx_max_r]))
    else: angle_r = 90
    ax_r.text(0.05, 0.95, f"Angle: {angle_r:.1f}°", transform=ax_r.transAxes, bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(GRAPH_PATH_TRAJECTORY)
    
    print(f"解析完了！グラフを保存しました: {GRAPH_PATH_TRAJECTORY}")
    plt.show()

if __name__ == "__main__":
    record_and_analyze_mm_fixed()
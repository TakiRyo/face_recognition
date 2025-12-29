# face recognition  

## eye_details.py. 
上瞼と下瞼がどれくらい動いているかを個別に測る。  
目の特徴長検出がノイジーなのか、測り方が良くないのか良い結果が出ない。  

## coner_of_mouth_detail.py. 
未実装
口角がどの方向にどれくらい上がっているか

## web_camera.py. 
web camera 撮影。
目尻の長さを使用してピクセルからmmに変換.  

## web camera select.py. 
目尻の長さか、シールか選べる。　　
params:  
eyebrow / eye / mouth threshold

## analyze_face.py  
水平を自動で調整して、結果を表示。時々ずれるけど簡単

## manual_face_analyze.py  
水平を手動で調整する。面倒だけど、精度は良い。

## manual_face_video.py  
web cameraを使って撮影できる。水平は写真撮影後に手動で水平調整　　

この上の三つは静止画像に対しての評価で、今はdiffrenceに符号がついているけど、これを絶対値で評価するべき。要変更


## excursion.py  
広角がどれくらい上がるか？  

## independence.py  
口を開けた時に目を開けたままでいられるか？  
import os
import json
import cv2
import numpy as np
from ultralytics import YOLO
import config
import torch # Import Ä‘á»ƒ cháº¯c cháº¯n vá» CUDA

# ==========================================
# 1. Cáº¤U HÃŒNH
# ==========================================
try:
    TASK1_DETECTION_CONFIG = config.return_task1_detection_config()
except AttributeError:
    TASK1_DETECTION_CONFIG = config.returnTestTask2_1_Config()

# Legacy alias used by app.py.
Task2_1Config = TASK1_DETECTION_CONFIG


# Kiá»ƒm tra key output trong config cá»§a báº¡n
OUTPUT_DIR = TASK1_DETECTION_CONFIG.get("output", TASK1_DETECTION_CONFIG.get("ouput"))

# --- Cáº¤U HÃŒNH VISUALIZE Má»šI ---
ENABLE_VISUALIZE = True # Báº­t/Táº¯t chá»©c nÄƒng visualize

# Táº¡o cÃ¡c thÆ° má»¥c output náº¿u chÆ°a cÃ³

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")

# ==========================================
# 2. HÃ€M Há»– TRá»¢ (Äá»ŒC/GHI áº¢NH VÃ€ CONVERT)
# ==========================================
# HÃ m Ä‘á»c áº£nh há»— trá»£ Ä‘Æ°á»ng dáº«n tiáº¿ng Viá»‡t/Windows
def read_image_windows(path):
    try:
        stream = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(stream, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None

# HÃ m lÆ°u áº£nh há»— trá»£ Ä‘Æ°á»ng dáº«n tiáº¿ng Viá»‡t/Windows
def save_image_windows(path, img):
    try:
        ext = os.path.splitext(path)[1]
        result, encoded_img = cv2.imencode(ext, img)
        if result:
            encoded_img.tofile(path)
            return True
    except Exception:
        pass
    return False

def convert_obb_to_json_structure(filename, obb_results):
    """Chuyá»ƒn Ä‘á»•i káº¿t quáº£ OBB cá»§a YOLO sang format JSON cá»§a bÃ i toÃ¡n vÃ  má»Ÿ rá»™ng OBB."""
    text_blocks = []
    
    # GiÃ¡ trá»‹ má»Ÿ rá»™ng (expansion)
    expansion = 2 
    
    if obb_results is not None:
        # Láº¥y tá»a Ä‘á»™ OBB (x0, y0, x1, y1, x2, y2, x3, y3)
        boxes = obb_results.xyxyxyxy.cpu().numpy()
        
        for idx, box in enumerate(boxes):
            # box lÃ  máº£ng 4x2 [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]
            
            # 1. TÃ­nh toÃ¡n tÃ¢m (Center) cá»§a OBB
            # TÃ¢m lÃ  trung bÃ¬nh cá»™ng cá»§a táº¥t cáº£ 4 Ä‘á»‰nh
            center_x = np.mean(box[:, 0])
            center_y = np.mean(box[:, 1])
            
            expanded_points = []
            for point in box:
                x_old, y_old = point[0], point[1]
                
                # 2. TÃ­nh toÃ¡n vector tá»« tÃ¢m Ä‘áº¿n Ä‘á»‰nh hiá»‡n táº¡i (dx, dy)
                dx = x_old - center_x
                dy = y_old - center_y
                
                # 3. TÃ­nh toÃ¡n Ä‘á»™ dÃ i cá»§a vector (khoáº£ng cÃ¡ch tá»« tÃ¢m)
                distance = np.sqrt(dx**2 + dy**2)
                
                if distance > 1e-6: # TrÃ¡nh chia cho 0 náº¿u box lÃ  má»™t Ä‘iá»ƒm (Ä‘iá»u kiá»‡n hiáº¿m gáº·p)
                    # 4. TÃ­nh toÃ¡n há»‡ sá»‘ má»Ÿ rá»™ng
                    # Äá»‰nh má»›i sáº½ náº±m trÃªn Ä‘Æ°á»ng tháº³ng Ä‘i qua tÃ¢m vÃ  Ä‘á»‰nh cÅ©, 
                    # nhÆ°ng xa tÃ¢m hÆ¡n má»™t khoáº£ng 'expansion'
                    
                    # Há»‡ sá»‘ tá»· lá»‡ má»›i: (Khoáº£ng cÃ¡ch cÅ© + expansion) / Khoáº£ng cÃ¡ch cÅ©
                    scale_factor = (distance + expansion) / distance
                    
                    # 5. TÃ­nh toÃ¡n tá»a Ä‘á»™ má»›i (má»Ÿ rá»™ng)
                    x_new = center_x + dx * scale_factor
                    y_new = center_y + dy * scale_factor
                else:
                    # Náº¿u box lÃ  1 Ä‘iá»ƒm, khÃ´ng thá»ƒ má»Ÿ rá»™ng theo cÃ¡ch nÃ y, giá»¯ nguyÃªn (hoáº·c xá»­ lÃ½ khÃ¡c tÃ¹y Ã½)
                    x_new, y_old = x_old, y_old 

                expanded_points.append([x_new, y_new])

            # Chuyá»ƒn list cÃ¡c Ä‘iá»ƒm má»Ÿ rá»™ng thÃ nh cáº¥u trÃºc JSON
            p = expanded_points
            polygon = {
                "x0": float(p[0][0]), "y0": float(p[0][1]),
                "x1": float(p[1][0]), "y1": float(p[1][1]),
                "x2": float(p[2][0]), "y2": float(p[2][1]),
                "x3": float(p[3][0]), "y3": float(p[3][1])
            }
            
            block = {"id": idx, "polygon": polygon}
            text_blocks.append(block)

    final_json = {
        "task1": {
            "input": {}, "name": "Chart Classification", "output": {"chart_type": "vertical bar"}
        },
        "task2": {
            "input": {"task1_output": {"chart_type": "vertical bar"}},
            "name": "Text Detection and Recognition",
            "output": {"text_blocks": text_blocks}
        }
    }
    return final_json

# --- HÃ€M VISUALIZE Má»šI ---
def visualize_obb(img_path, json_data, output_dir):
    """Váº½ bounding box tá»« JSON lÃªn áº£nh gá»‘c"""
    img = read_image_windows(img_path)
    if img is None: return

    blocks = json_data["task2"]["output"]["text_blocks"]
    for block in blocks:
        poly = block["polygon"]
        # Láº¥y 4 Ä‘iá»ƒm cá»§a Ä‘a giÃ¡c
        pts = np.array([
            [poly["x0"], poly["y0"]],
            [poly["x1"], poly["y1"]],
            [poly["x2"], poly["y2"]],
            [poly["x3"], poly["y3"]]
        ], np.int32)
        pts = pts.reshape((-1, 1, 2))

        # Váº½ Ä‘a giÃ¡c mÃ u xanh lÃ¡ (Green), Ä‘á»™ dÃ y 2
        cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    # LÆ°u áº£nh
    filename = os.path.basename(img_path)
    save_path = os.path.join(output_dir, filename)
    save_image_windows(save_path, img)

# ==========================================
# 3. MAIN INFERENCE
# ==========================================
def main():
    cfg = Task2_1Config if isinstance(Task2_1Config, dict) else TASK1_DETECTION_CONFIG

    MODEL_PATH = cfg["weight"]
    INPUT_IMG_DIR = cfg["input"]
    OUTPUT_DIR = cfg.get("output", cfg.get("ouput"))
    OUTPUT_JSON_DIR = OUTPUT_DIR
    OUTPUT_VIS_DIR = os.path.join(OUTPUT_DIR, "visualized_images")
    if not os.path.exists(OUTPUT_JSON_DIR): os.makedirs(OUTPUT_JSON_DIR)
    if ENABLE_VISUALIZE and not os.path.exists(OUTPUT_VIS_DIR): os.makedirs(OUTPUT_VIS_DIR)
    print(f"--- Äang load model YOLO tá»«: {MODEL_PATH} ---")
    # Kiá»ƒm tra CUDA cho RTX 3050
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Inference device: {device} ({torch.cuda.get_device_name(0) if device == 0 else 'CPU'})")
    
    model = YOLO(MODEL_PATH)
    
    files = [f for f in os.listdir(INPUT_IMG_DIR) if f.lower().endswith(VALID_EXTENSIONS)]
    total = len(files)
    print(f"TÃ¬m tháº¥y {total} áº£nh. Báº¯t Ä‘áº§u xá»­ lÃ½...")
    
    for idx, filename in enumerate(files):
        img_path = os.path.join(INPUT_IMG_DIR, filename)
        json_name = os.path.splitext(filename)[0] + ".json"
        save_json_path = os.path.join(OUTPUT_JSON_DIR, json_name)
        
        print(f"[{idx+1}/{total}] Detecting: {filename} ...", end="\r")
        
        try:
            # 1. Run Inference (trÃªn GPU, imgsz lá»›n cho text nhá»)
            results = model.predict(img_path, save=False, conf=0.25, verbose=False, device=device, imgsz=1024)
            result = results[0]
            
            # 2. Táº¡o dá»¯ liá»‡u JSON
            if result.obb is not None:
                json_data = convert_obb_to_json_structure(filename, result.obb)
            else:
                json_data = convert_obb_to_json_structure(filename, None)
            
            # 3. LÆ°u file JSON
            with open(save_json_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, ensure_ascii=False, indent=4)
                
            # 4. VISUALIZE Káº¾T QUáº¢ (Má»šI)
            if ENABLE_VISUALIZE:
                visualize_obb(img_path, json_data, OUTPUT_VIS_DIR)
                
        except Exception as e:
            print(f"\n[ERR] Lá»—i khi xá»­ lÃ½ áº£nh {filename}: {e}")

    print(f"\n\n[DONE] HoÃ n táº¥t!")
    print(f"- File JSON lÆ°u táº¡i: {OUTPUT_JSON_DIR}")
    if ENABLE_VISUALIZE:
        print(f"- áº¢nh Visualize lÆ°u táº¡i: {OUTPUT_VIS_DIR}")


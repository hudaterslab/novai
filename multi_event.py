import os
import sys
import gc
import json
import csv
import cv2
import math
import numpy as np
import time
import datetime
import traceback
import threading
import queue
import logging
import psutil
import atexit
from collections import deque, defaultdict
import concurrent.futures
import re
import requests
import pytz
from urllib.parse import urlsplit, unquote
from logging.handlers import TimedRotatingFileHandler, QueueHandler, QueueListener
import argparse

warnings = requests.packages.urllib3.exceptions.InsecureRequestWarning
requests.packages.urllib3.disable_warnings(warnings)

# ==========================================
# [1] 시스템 기본 설정 및 상수
# ==========================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_COMMON_FILE = os.path.join(PROJECT_ROOT, "system_config.json")
CONFIG_CAMERAS_FILE = os.path.join(PROJECT_ROOT, "cameras.json")
CAMERA_LIST_FILE = os.path.join(PROJECT_ROOT, "cameras.csv")
EVENT_ROOT_DIR = os.path.join(PROJECT_ROOT, "CCTV_EVENT_ALERT")

SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
WATCHDOG_TIMEOUT = 30.0

# YOLOv8 클래스 ID 정의 (hanjin_cctv.dxnn 기준)
ID_H_HELMET = 0
ID_H_NO_HELMET = 1
ID_G_PERSON = 2
ID_G_CAR = 3
ID_PERSON_LOW = 4
ID_REFLECTIVE_VEST = 5
ID_G_TRUCK = 6
TARGET_VEHICLES = [ID_G_CAR, ID_G_TRUCK]

DEBUG_MODE = False

# ------------------------------------------------------------
# ROI 보정(Aligner) 튜닝 파라미터 
# ------------------------------------------------------------
ALIGN_INTERVAL_SEC = 300.0
ORB_FEATURES = 1500
MIN_GOOD_MATCHES = 20
MIN_INLIERS = 12
MIN_INLIER_RATIO = 0.25
RANSAC_REPROJ_THRESH = 5.0
TRACKING_UPDATE_MIN_INTERVAL_SEC = 2.0
TRACKING_UPDATE_MIN_INLIERS = 25
TRACKING_UPDATE_MIN_INLIER_RATIO = 0.35
ANCHOR_DIRECT_CHECK_INTERVAL_SEC = 15.0
ANCHOR_DIRECT_MIN_INLIERS = 30
ANCHOR_DIRECT_MIN_INLIER_RATIO = 0.35
MAX_CORNER_SHIFT_RATIO = 0.45      
MAX_SCALE_CHANGE = 0.45            
MAX_PERSPECTIVE_ABS = 0.003        
HOMOGRAPHY_IDENTITY_ATOL = 1e-3
ROI_APPLY_MIN_SHIFT_PX = 5.0

MIN_APPLY_TRANSLATION_PX = 5.0     
MIN_APPLY_ROTATION_DEG = 0.5       
MIN_APPLY_SCALE_CHANGE = 0.02      
MIN_APPLY_PERSPECTIVE = 0.0005     
KEEP_LAST_GOOD_ROI_ON_FAILURE = True
DEBUG_ALIGN = True

def deep_merge_dict(base, override):
    """딕셔너리를 깊은 병합(Deep Merge)하는 유틸리티 함수"""
    import copy
    result = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = deep_merge_dict(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result

def load_system_config():
    default_config = {
        "terminal_id": "99999",
        "logging": {"dir": "./logs", "level": "INFO"},
        "event_config": {
            "intrusion": {"enabled": False, "cooldown_sec": 600},
            "illegal_parking": {"enabled": False, "cooldown_sec": 600, "trigger_sec": 5.0, "move_threshold_ratio": 0.1},
            "no_helmet": {"enabled": False, "cooldown_sec": 600, "blur_face": True, "trigger_sec": 3.0},
            "conveyor_crossing": {
                "enabled": False, "cooldown_sec": 600, "snapshot_mode": "crossing_moment", 
                "distance_ratio": 0.9, "min_crossing_angle": 20.0, "candidate_ttl_sec": 5.0
            },
            "signal_vehicle": {"enabled": False, "cooldown_sec": 600, "motion_threshold_ratio": 0.30}
        },
        "models": {
            "MAIN": "hanjin_cctv.dxnn",
            "FACE": "yolov8m-face.dxnn",
            "HELMET": "helmet_3cls_v8.dxnn"
        },
        "model_confidences": {
            "MAIN": 0.6,
            "FACE": 0.35,
            "HELMET": 0.55,
            "PERSON": 0.35  # [추가] 사람 및 신호수 전용 기본 임계값 설정
        },
        "BATCH_SIZE": 9,
        "REC_FPS": 3,
        "LOOP_FPS": 15,
        "REC_PRE_SEC": 10,
        "REC_POST_SEC": 10,
        "VISUAL_ALARM_DURATION": 5.0
    }
    
    if not os.path.exists(CONFIG_COMMON_FILE):
        return default_config
        
    try:
        with open(CONFIG_COMMON_FILE, 'r', encoding='utf-8') as f:
            loaded_config = json.load(f)
            return deep_merge_dict(default_config, loaded_config)
    except Exception as e:
        print(f"[Warning] 설정 파일 로드 실패. 기본값을 사용합니다: {e}")
        return default_config

SYS_CFG = load_system_config()
BATCH_SIZE = SYS_CFG.get("BATCH_SIZE", 9)
IMAGE_SAVER_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# ==========================================
# [2] 로깅 시스템 초기화
# ==========================================
LOG_DIR = SYS_CFG.get("logging", {}).get("dir", "./logs")
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("CCTV_SYSTEM")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)-7s | [%(funcName)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

log_filename = datetime.datetime.now().strftime("cctv_%Y%m%d.log")
log_filepath = os.path.join(LOG_DIR, log_filename)

file_handler = TimedRotatingFileHandler(log_filepath, when="H", interval=1, backupCount=24, encoding='utf-8')
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)

# 비동기 로깅을 위한 큐(Queue) 설정
log_queue = queue.Queue(-1)
queue_handler = QueueHandler(log_queue)
logger.addHandler(queue_handler)

LOG_LISTENER = QueueListener(log_queue, file_handler, stream_handler, respect_handler_level=True)
LOG_LISTENER.start()

def graceful_shutdown():
    """시스템 종료 시 스레드 풀과 로거를 안전하게 정리합니다."""
    if LOG_LISTENER is not None:
        try:
            LOG_LISTENER.stop()
        except Exception:
            pass
    print("🔄 [SYSTEM] 백그라운드 I/O 작업을 안전하게 마무리하고 종료합니다...")
    try:
        IMAGE_SAVER_POOL.shutdown(wait=True)
    except Exception:
        pass

atexit.register(graceful_shutdown)

# ==========================================
# [3] 딥엑스 NPU 엔진 및 환경변수 설정
# ==========================================
# ==========================================
# [3] 딥엑스 NPU 엔진 및 환경변수 설정
# ==========================================
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["OPENCV_LOG_LEVEL"] = "FATAL"
os.environ["OPENCV_FFMPEG_DEBUG"] = "0"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;3000000|max_delay;500000"

# [수정] sys.exit(1) 강제 종료를 제거하고 상태 플래그(HAS_DX_ENGINE) 도입
HAS_DX_ENGINE = False
try:
    from dx_engine import InferenceEngine, InferenceOption
    HAS_DX_ENGINE = True
except ImportError:
    logger.warning("💡 [환경 알림] dx_engine 모듈을 찾을 수 없습니다. 서버(GPU/CPU) 환경으로 간주합니다.")

# ==========================================
# [4] 공통 유틸리티 함수
# ==========================================
def sanitize_camera_url(url: str) -> str:
    """RTSP URL에서 공백 및 불순물을 제거합니다."""
    if not url: return ""
    try:
        clean_url = url.encode('ascii', 'ignore').decode('ascii')
        return re.sub(r'\s+', '', clean_url.strip())
    except Exception:
        return re.sub(r'\s+', '', str(url).strip())

def extract_ip(rtsp_url: str) -> str:
    """RTSP URL에서 식별용 고유 ID(IP+Channel)를 추출합니다."""
    try:
        clean_url = sanitize_camera_url(rtsp_url)
        if "://" not in clean_url:
            clean_url = f"rtsp://{clean_url}"
            
        parsed = urlsplit(clean_url)
        host = parsed.netloc.rsplit("@", 1)[-1].strip("[]").split(":")[0].split(".")[-1]
        
        # Path와 Query(채널 정보 등)를 포함하여 고유한 키 생성
        path = re.sub(r'[^a-zA-Z0-9]', '_', parsed.path)
        query = re.sub(r'[^a-zA-Z0-9]', '_', parsed.query)
        
        uid = f"{host}{path}_{query}".strip('_')
        return uid if uid else "unknown_cam"
    except Exception as e:
        logger.warning(f"고유 식별자 추출 실패: {e}")
        return "unknown_cam"

def load_rtsp_list_from_csv(csv_path):
    """CSV 파일에서 카메라 RTSP URL 목록을 로드합니다."""
    if not os.path.exists(csv_path):
        logger.error(f"카메라 목록 CSV를 찾을 수 없습니다: {csv_path}")
        return []

    rtsp_list = []
    try:
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'): 
                    continue
                first_col = line.split(',')[0].strip()
                if first_col.lower() in ['url', 'rtsp', 'rtsp_url', 'camera_url']:
                    continue
                url = sanitize_camera_url(first_col)
                if url: 
                    rtsp_list.append(url)
    except Exception as e: 
        logger.error(f"카메라 리스트 로드 중 예외 발생: {e}")
        pass
        
    unique_list = []
    for u in rtsp_list:
        if u not in unique_list: 
            unique_list.append(u)
            
    logger.info(f"카메라 CSV 로드 완료: {len(unique_list)}대")
    return unique_list

def calculate_iou(box1, box2):
    """두 BBox 간의 IoU(Intersection over Union)를 계산합니다."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0: 
        return 0
        
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    return inter_area / (box1_area + box2_area - inter_area)

def get_foot_point(x1, y1, x2, y2): 
    return (int((x1 + x2) / 2), int(y1 + (y2 - y1) * (2/3)))

def get_check_point(x1, y1, x2, y2): 
    return (int((x1 + x2) / 2), int(y2))

def get_center_point(x1, y1, x2, y2): 
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

def get_distance(p1, p2): 
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def ccw(p1, p2, p3):
    """세 점의 방향성을 판별합니다. (선분 교차 알고리즘용)"""
    val = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
    if val > 0: return 1
    elif val < 0: return -1
    return 0

def normalize_roi_points(points, width, height):
    if not points or width <= 0 or height <= 0:
        return []
    return [[round(float(x) / width, 6), round(float(y) / height, 6)] for x, y in points]

def denormalize_roi_points(points, width, height):
    if not points or width <= 0 or height <= 0:
        return []
    return [[int(round(float(x) * width)), int(round(float(y) * height))] for x, y in points]

def create_mosaic_image(images, screen_w=SCREEN_WIDTH, screen_h=SCREEN_HEIGHT):
    """여러 카메라의 영상을 하나의 모자이크 화면으로 합성합니다."""
    if not images: 
        return None
        
    count = len(images)
    cols = max(1, math.ceil(math.sqrt(count)))
    rows = max(1, math.ceil(count / cols))
    
    cell_w = screen_w // cols
    cell_h = screen_h // rows
    
    mosaic = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
    
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        x, y = c * cell_w, r * cell_h
        
        if img is None:
            cell_img = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
            cv2.putText(cell_img, "No Signal", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else: 
            cell_img = cv2.resize(img, (cell_w, cell_h))
            
        mosaic[y:y+cell_h, x:x+cell_w] = cell_img
        cv2.rectangle(mosaic, (x, y), (x+cell_w, y+cell_h), (100, 100, 100), 1)
        
    return mosaic

# ==========================================
# [5] API 통신 및 이미지 저장 (NAS 연동 제외)
# ==========================================
def send_event_image_to_receiver(image_path, event_name, terminal_id, cctv_id, bboxes, img_width=None, img_height=None):
    """수신 서버(Receiver API)로 이벤트 이미지를 POST 전송합니다."""
    if(terminal_id == "99999"):
        logger.debug(f"[API 스킵] 기본 단말 ID(99999) 사용 중: {image_path}")
        return
    
    url = "https://tmlsafety.hudaters.net/receiver/api/v1/cctv/img"
    event_type_mapping = {
        "conveyor_crossing": 1, 
        "no_helmet": 2, 
        "signal_vehicle": 3, 
        "illegal_parking": 4, 
        "intrusion": 5
    }
    
    if event_name not in event_type_mapping:
        logger.debug(f"[API 스킵] 정의되지 않은 이벤트 타입: {event_name}")
        return
        
    api_event_type = event_type_mapping[event_name]
    kst = pytz.timezone('Asia/Seoul')
    collected_at = datetime.datetime.now(kst).strftime('%Y-%m-%dT%H:%M:%S')
    
    # [검증 완료] bboxes 배열(리스트 내 딕셔너리)을 JSON 문자열로 안전하게 직렬화
    detected_objects_json = json.dumps(bboxes) if bboxes else "[]"

    data = {
        "collectedAt": collected_at,
        "eventType": api_event_type,
        "terminalId": str(terminal_id),
        "cctvId": int(cctv_id),
        "detectedObjects": detected_objects_json
    }
    
    if img_width: data["imageWidth"] = int(img_width)
    if img_height: data["imageHeight"] = int(img_height)

    if not os.path.exists(image_path):
        logger.error(f"[API 에러] 파일을 찾을 수 없습니다: {image_path}")
        return

    try:
        with open(image_path, 'rb') as f:
            files = {"image": (os.path.basename(image_path), f, "image/jpeg")}
            response = requests.post(url, data=data, files=files, verify=False, timeout=10)
            
            if response.status_code == 200:
                logger.info(f"🌐 [API 전송 성공] 단말:{terminal_id} | CAM:{cctv_id} | 이벤트:{event_name}")
            else:
                logger.error(f"⚠️ [API 전송 실패] 상태코드: {response.status_code} | 메시지: {response.text}")
    except requests.exceptions.RequestException as e:
        logger.error(f"⚠️ [API 네트워크 예외 발생]: {e}")
    except Exception as e:
        logger.error(f"⚠️ [API 기타 예외 발생]: {e}\n{traceback.format_exc()}")

def _save_and_send_task(img, img_path, api_params):
    """비동기 스레드에서 파일 쓰기 및 API 전송을 처리합니다."""
    try:
        cv2.imwrite(img_path, img)
    except Exception as e:
        logger.error(f"[이미지 저장 실패] 경로: {img_path} | 예외: {e}")
        return
        
    try:
        send_event_image_to_receiver(
            image_path=img_path,
            event_name=api_params['event_name'],
            terminal_id=api_params['terminal_id'],
            cctv_id=api_params['cctv_id'],
            bboxes=api_params['bboxes'],
            img_width=api_params['img_width'],
            img_height=api_params['img_height']
        )
    except Exception as e:
        logger.error(f"[Task 내부 API 호출 에러] {e}")

def save_event_image_with_mark(frame, ip, event_type, bbox, tid, terminal_id="99999", cctv_id=1, objects_meta=None, trajectories=None, auth_tokens=None):
    """프레임에 BBox, 다중 궤적, 신호수 토큰 적용 절대 시간을 마킹하고 이미지를 로컬에 저장한 후 API 큐에 등록합니다."""
    if IMAGE_SAVER_POOL._work_queue.qsize() > 50:
        logger.warning("이미지 저장 큐가 포화 상태입니다. 저장을 스킵합니다.")
        return
        
    try:
        img = frame.copy()
        x1, y1, x2, y2 = map(int, bbox)
        
        # 1. 다중 객체 궤적(Trajectory) 렌더링
        #if trajectories:
        #    for obj_tid, hist_pts in trajectories.items():
        #        if len(hist_pts) > 1:
                    # [수정] 궤적 굵기를 3에서 1로 얇게 수정
        #            cv2.polylines(img, [np.array(hist_pts, np.int32)], False, (255, 0, 255), 1, cv2.LINE_AA)
        #            cv2.circle(img, hist_pts[0], 3, (0, 255, 255), -1)
        #            cv2.circle(img, hist_pts[-1], 4, (0, 0, 255), -1)
                    
        # 2. 메인 BBox 및 이벤트 정보 텍스트 렌더링
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        now = datetime.datetime.now()
        msg = f"{event_type} ID:{tid} {now.strftime('%H:%M:%S')}"
        text_y = max(20, y1 - 10)
        cv2.putText(img, msg, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 3. 신호수 유예 토큰 상태 렌더링 - [img_local 에만 엄격히 제한]
        if event_type == "signal_vehicle":
            safe_tokens = auth_tokens if auth_tokens else []
            target_token = next((t for t in safe_tokens if t['tid'] == tid), None)
            
            h_img, w_img = img_local.shape[:2]
            box_w = 340
            box_h = 75 
            x_start = w_img - box_w - 20
            y_start = h_img - box_h - 20

            overlay = img_local.copy()
            cv2.rectangle(overlay, (x_start, y_start), (x_start + box_w, y_start + box_h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, img_local, 0.4, 0, img_local)

            cv2.putText(img_local, "Signalman Auth", (x_start + 10, y_start + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            if target_token:
                # [복구] API 로컬 이미지도 Truck ID 제외하고 남은 시간 표출
                auth_time_str = datetime.datetime.fromtimestamp(target_token['auth_t']).strftime('%H:%M:%S')
                sig_id = target_token.get('sig_tid', 'Unknown')
                remain_time = target_token.get('remain', 0.0)
                cv2.putText(img_local, f"Auth: {auth_time_str} | Remain: {remain_time:.1f}s", (x_start + 10, y_start + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(img_local, f"Authorized by: Signalman [{sig_id}]", (x_start + 10, y_start + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            else:
                # [수정] 알람이 발생해서 API 이미지가 저장되는 시점이므로 여기서는 UNAUTH 표출이 맞음
                cv2.putText(img_local, "Status: UNAUTH (ALARM)", (x_start + 10, y_start + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(img_local, "Reason: Moving without Signalman", (x_start + 10, y_start + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
                
        # 4. 저장 경로 생성 및 비동기 API 전송 태스크 큐 등록
        dpath = os.path.join(EVENT_ROOT_DIR, "events", ip, "images", str(event_type))
        if not os.path.exists(dpath):
            os.makedirs(dpath, exist_ok=True)
            
        fname = f"{now.strftime('%Y%m%d_%H%M%S')}_{ip}_{event_type}_{tid}.jpg"
        img_path = os.path.join(dpath, fname)
        
        h, w = frame.shape[:2]
        
        if objects_meta:
            ai_detected_bboxes = [
                {
                    "box": [int(b) for b in o['box']], 
                    "label": str(o['label']), 
                    "score": round(float(o.get('score', 0.95)), 2)
                } 
                for o in objects_meta
            ]
        else:
            ai_detected_bboxes = [
                {
                    "box": [x1, y1, x2, y2], 
                    "label": str(event_type), 
                    "score": 0.95
                }
            ]
        
        api_params = {
            'ip': ip,
            'event_name': event_type,
            'terminal_id': str(terminal_id),
            'cctv_id': int(cctv_id),            
            'bboxes': ai_detected_bboxes,
            'img_width': w,
            'img_height': h
        }
        
        IMAGE_SAVER_POOL.submit(_save_and_send_task, img, img_path, api_params)
        
    except Exception as e: 
        logger.error(f"[EventLogic Error] 이미지 마킹 중 예외 발생: {e}")

# ==========================================
# [6] DeepX NPU 모델 추론 (YOLOv8 버그 픽스 반영)
# ==========================================
class YoLoDeepX:
    def __init__(self, engine_path):
        # [수정] 객체 생성 시점에 NPU 환경인지 체크하여 안전하게 방어
        if not HAS_DX_ENGINE:
            raise RuntimeError("dx_engine이 설치되지 않은 서버/PC 환경에서는 YoLoDeepX(NPU) 객체를 생성할 수 없습니다.")
            
        self.engine_path = engine_path
        try:
            io = InferenceOption()
            self.engine = InferenceEngine(self.engine_path, io)
            logger.info(f"[DeepX] 모델 로드 성공: {os.path.basename(self.engine_path)}")
        except Exception as e:
            logger.error(f"[DeepX Load Fail] 엔진 초기화 실패 ({engine_path}): {e}")
            raise e

    def letter_box(self, img, new_shape=(640,640)):
        h, w = img.shape[:2]
        scale = min(new_shape[0]/h, new_shape[1]/w)
        nw, nh = int(w*scale), int(h*scale)
        
        resized = cv2.resize(img, (nw, nh))
        canvas = np.full((new_shape[0], new_shape[1], 3), 114, dtype=np.uint8)
        
        dw, dh = (new_shape[1] - nw) // 2, (new_shape[0] - nh) // 2
        canvas[dh:dh+nh, dw:dw+nw] = resized
        
        return canvas, scale, (dw, dh)

    def postprocess(self, output_tensor, conf_thres=0.40, iou_thres=0.45):
        try:
            pred = np.array(output_tensor[0])
            
            # YOLOv8 배열 형태 보정
            if pred.ndim == 3 and pred.shape[1] < pred.shape[2]: 
                pred = pred.transpose((0, 2, 1))
            if pred.ndim == 3: 
                pred = pred[0]
            
            # Class-Id 및 Score 추출
            scores = np.max(pred[:, 4:], axis=1)
            class_ids = np.argmax(pred[:, 4:], axis=1)

            # Confidence 필터링
            mask = scores > conf_thres
            pred = pred[mask]
            scores = scores[mask]
            class_ids = class_ids[mask]
            
            if len(pred) == 0: 
                return []

            # NMSBoxes 포맷 맞춤 (x_min, y_min, width, height)
            boxes_xywh = pred[:, :4].copy()
            boxes_xywh[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2  # 중심 X -> 최소 X
            boxes_xywh[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2  # 중심 Y -> 최소 Y
            
            # Class-Aware NMS
            max_wh = 7680 
            class_offset = class_ids * max_wh
            boxes_shifted = boxes_xywh.copy()
            boxes_shifted[:, 0] += class_offset
            boxes_shifted[:, 1] += class_offset
            
            indices = cv2.dnn.NMSBoxes(boxes_shifted.tolist(), scores.tolist(), conf_thres, iou_thres)
            
            results = []
            if len(indices) > 0:
                for i in indices.flatten():
                    x_min, y_min, w, h = boxes_xywh[i]
                    results.append([[x_min, y_min, x_min + w, y_min + h], scores[i], class_ids[i]])
                    
            return results
        except Exception as e:
            logger.error(f"NPU Postprocess Error ({os.path.basename(self.engine_path)}): {e}")
            return []

    def infer(self, img, conf_override=None):
        if img is None: 
            return np.empty((0,6))
            
        h_orig, w_orig = img.shape[:2]
        npu_input, scale, offset = self.letter_box(img)
        npu_input_rgb = cv2.cvtColor(npu_input, cv2.COLOR_BGR2RGB)
        
        try:
            output_tensor = self.engine.run([npu_input_rgb])
            
            thres = conf_override if conf_override is not None else 0.40
            raw_dets = self.postprocess(output_tensor, conf_thres=thres)
            
            if not raw_dets: 
                return np.empty((0,6))
            
            res = []
            dw, dh = offset
            
            for box, score, cls_id in raw_dets:
                x1 = np.clip((box[0] - dw) / scale, 0, w_orig)
                y1 = np.clip((box[1] - dh) / scale, 0, h_orig)
                x2 = np.clip((box[2] - dw) / scale, 0, w_orig)
                y2 = np.clip((box[3] - dh) / scale, 0, h_orig)
                
                res.append([x1, y1, x2, y2, score, cls_id])
                
            return np.array(res)
        except Exception as e:
            logger.error(f"NPU Inference Error: {e}")
            return np.empty((0,6))

# ==========================================
# [7] 객체 트래커 및 영상 녹화기
# ==========================================
# [1] 시스템 기본 설정 및 상수 영역 하단에 추가
DEBUG_MODE = False
class SimpleTracker:
    def __init__(self, max_lost=30, history_len=60): # 15FPS 기준 약 4초 분량의 궤적 저장
        self.next_id = 1
        self.tracks = {}
        self.max_lost = max_lost
        self.history_len = history_len

    def update(self, detections):
        used_dets = set()
        
        for tid, trk in self.tracks.items():
            best_iou = 0
            best_idx = -1
            
            for i, det in enumerate(detections):
                if i in used_dets: 
                    continue
                if int(det[5]) != trk['cls']: 
                    continue
                    
                iou = calculate_iou(trk['bbox'], det[:4])
                if iou > best_iou: 
                    best_iou = iou
                    best_idx = i
                    
            if best_iou > 0.2:
                # 중심점 좌표 계산 및 히스토리에 누적
                cx = int((detections[best_idx][0] + detections[best_idx][2]) / 2)
                cy = int((detections[best_idx][1] + detections[best_idx][3]) / 2)
                
                self.tracks[tid].update({
                    'bbox': detections[best_idx][:4], 
                    'lost': 0, 
                    'conf': detections[best_idx][4]
                })
                self.tracks[tid]['history'].append((cx, cy))
                used_dets.add(best_idx)
            else: 
                self.tracks[tid]['lost'] += 1
                
        self.tracks = {tid: t for tid, t in self.tracks.items() if t['lost'] <= self.max_lost}
        
        res_tracks = []
        for i, det in enumerate(detections):
            if i not in used_dets:
                cx = int((det[0] + det[2]) / 2)
                cy = int((det[1] + det[3]) / 2)
                self.tracks[self.next_id] = {
                    'bbox': det[:4], 'lost': 0, 'cls': int(det[5]), 'conf': det[4],
                    'history': deque([(cx, cy)], maxlen=self.history_len) # 신규 객체 궤적 초기화
                }
                self.next_id += 1
                
        for tid, trk in self.tracks.items():
            if trk['lost'] == 0:
                res_tracks.append([*trk['bbox'], tid, trk.get('conf', 1.0), trk['cls']])
                
        return np.array(res_tracks)

class VideoRecorder:
    def __init__(self, ip):
        self.ip = ip
        self.fps = SYS_CFG.get("REC_FPS", 3)
        self.buffer = deque(maxlen=self.fps * SYS_CFG.get("REC_PRE_SEC", 10))
        self.write_queue = queue.Queue()
        
        self.recording = False
        self.record_end_time = 0
        self.current_event = "unknown"
        self.running = True
        
        self.thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.thread.start()

    def update(self, frame):
        if frame is None: 
            return
            
        self.buffer.append(frame)
        
        if self.recording:
            if time.time() > self.record_end_time:
                self.recording = False
                self.write_queue.put(None)
                logger.info(f"🎬 [녹화종료] {self.ip} - {self.current_event}")
            else:
                self.write_queue.put(frame)

    def trigger(self, event_name, objects_meta=None): # [수정] objects_meta 매개변수 추가
        now = time.time()
        post_sec = SYS_CFG.get("REC_POST_SEC", 10)
        
        if self.recording:
            self.record_end_time = now + post_sec
        else:
            logger.info(f"🎥 [녹화시작] {self.ip} - {event_name}")
            self.recording = True
            self.record_end_time = now + post_sec
            self.current_event = event_name
            self.current_meta = objects_meta # [추가] 이벤트 발생 시점의 BBox 메타데이터 기억
            
            temp_buffer = list(self.buffer)
            for f in temp_buffer: 
                self.write_queue.put(f)

    def _writer_loop(self):
        writer = None
        while self.running:
            try:
                frame = self.write_queue.get(timeout=1.0)
            except queue.Empty: 
                continue

            if frame is None:
                if writer: 
                    writer.release()
                    writer = None
                continue

            if writer is None:
                dpath = os.path.join(EVENT_ROOT_DIR, "events", self.ip, "videos", self.current_event)
                if not os.path.exists(dpath): 
                    os.makedirs(dpath, exist_ok=True)
                    
                # 파일명 동기화를 위해 변수 처리
                time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                fname = f"{time_str}_{self.ip}_{self.current_event}.mp4"
                fpath = os.path.join(dpath, fname)
                
                # -----------------------------------------------------------
                # [추가] 영상 생성 시점에 BBox 상세 수치 데이터(JSON)를 동시 저장
                # -----------------------------------------------------------
                if hasattr(self, 'current_meta') and self.current_meta:
                    meta_fname = f"{time_str}_{self.ip}_{self.current_event}.json"
                    meta_path = os.path.join(dpath, meta_fname)
                    try:
                        with open(meta_path, 'w', encoding='utf-8') as f_meta:
                            json.dump(self.current_meta, f_meta, indent=4, ensure_ascii=False)
                        logger.info(f"📝 [BBox 데이터 저장 완료] 경로: {meta_path}")
                    except Exception as e:
                        logger.error(f"⚠️ BBox JSON 메타데이터 저장 실패: {e}")
                # -----------------------------------------------------------
                
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(fpath, fourcc, self.fps, (w, h))
                
                if not writer.isOpened():
                    logger.error(f"[녹화에러] 파일을 열 수 없습니다: {fpath}")
                    writer = None
                    continue
                    
            if writer: 
                writer.write(frame)

class MotionDetector:
    def __init__(self, sensitivity=5):
        self.threshold = 100 - ((sensitivity - 1) * 9)
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=self.threshold, detectShadows=True)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
    def apply(self, frame):
        if frame is None: 
            return None
        small_frame = cv2.resize(frame, (640, 360))
        fg_mask = self.bg_subtractor.apply(small_frame)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
        return fg_mask

# ==========================================
# [8] 정밀 이벤트 감지 로직 (OOP 구조)
# ==========================================
class BaseEventDetector:
    gui_name = "BASE"
    def __init__(self, config, roi_poly=None, roi_lines=None):
        self.config = config
        self.roi_poly = np.array(roi_poly, dtype=np.int32) if roi_poly and len(roi_poly) >= 3 else np.empty((0, 2), dtype=np.int32)
        self.roi_lines = roi_lines or []
        self.fps = SYS_CFG.get("REC_FPS", 3)
        
    def process(self, tracks, track_map, motion_mask, frame, fid, **kwargs): 
        return []

class IntrusionDetector(BaseEventDetector):
    gui_name = "INTRUSION"
    
    def process(self, tracks, track_map, motion_mask, frame, fid, **kwargs):
        triggered = []
        if self.roi_poly.size == 0: 
            return triggered
            
        for t in tracks:
            tid = int(t[4])
            if track_map.get(tid) == ID_G_PERSON:
                if cv2.pointPolygonTest(self.roi_poly, get_foot_point(*t[:4]), False) >= 0:
                    triggered.append({
                        'tid': tid, 
                        'bbox': t[:4], 
                        'frame': frame, 
                        'fid': fid
                    })
                    
        return triggered

class ParkingDetector(BaseEventDetector):
    gui_name = "PARKING"
    
    def __init__(self, config, roi_poly=None, roi_lines=None):
        super().__init__(config, roi_poly, roi_lines)
        self.states = defaultdict(lambda: {'start_time': 0.0, 'pos': None})
        
        self.trigger_sec = config.get("trigger_sec", 5.0)
        self.move_threshold_ratio = config.get("move_threshold_ratio", 0.1)
        
    def process(self, tracks, track_map, motion_mask, frame, fid, **kwargs):
        triggered = []
        curr_ids = set()
        current_time = time.time()
        
        if self.roi_poly.size == 0: 
            return triggered
            
        for t in tracks:
            tid = int(t[4])
            if track_map.get(tid) in TARGET_VEHICLES:
                if cv2.pointPolygonTest(self.roi_poly, get_check_point(*t[:4]), False) >= 0:
                    curr_ids.add(tid)
                    
                    x1, y1, x2, y2 = t[:4]
                    c = get_center_point(x1, y1, x2, y2)
                    vehicle_size = max(x2 - x1, y2 - y1)
                    
                    dynamic_move_threshold = vehicle_size * self.move_threshold_ratio
                    
                    if self.states[tid]['start_time'] == 0.0 or get_distance(c, self.states[tid]['pos']) > dynamic_move_threshold:
                        self.states[tid].update({
                            'start_time': current_time, 
                            'pos': c, 
                            'bbox': t[:4], 
                            'frame': frame.copy() if frame is not None else None,
                            'fid': fid,
                            'triggered': False
                        })
                    else:
                        self.states[tid].update({
                            'bbox': t[:4],
                            'frame': frame.copy() if frame is not None else None,
                            'fid': fid
                        })
                        
                        duration_sec = current_time - self.states[tid]['start_time']
                        
                        if not self.states[tid].get('triggered', False) and duration_sec >= self.trigger_sec:
                            triggered.append({
                                'tid': tid, 
                                'bbox': self.states[tid]['bbox'], 
                                'frame': self.states[tid]['frame'],
                                'fid': self.states[tid]['fid']
                            })
                            self.states[tid]['triggered'] = True
                        
        for tid in list(self.states.keys()):
            if tid not in curr_ids: 
                del self.states[tid]
                
        return triggered

class CrossingDetector(BaseEventDetector):
    gui_name = "CROSSING"
    
    def __init__(self, config, roi_poly=None, roi_lines=None):
        super().__init__(config, roi_poly, roi_lines)
        self.lines = []
        for i in range(0, len(self.roi_lines), 2):
            if i + 1 < len(self.roi_lines):
                self.lines.append((self.roi_lines[i], self.roi_lines[i+1]))
                
        self.prev = {}
        self.candidates = {}
        self.lb_offsets = {}
        self.lb_last_height = {}
        
        self.min_crossing_angle = config.get("min_crossing_angle", 20.0)
        self.distance_ratio = config.get("distance_ratio", 0.2)
        
        self.candidate_ttl_sec = config.get("candidate_ttl_sec", 5.0)

    def _is_intersect(self, p1, p2, p3, p4): 
        c1 = ccw(p1, p2, p3) * ccw(p1, p2, p4)
        c2 = ccw(p3, p4, p1) * ccw(p3, p4, p2)
        return c1 <= 0 and c2 <= 0
        
    def _get_perpendicular_distance(self, p1, p2, pt):
        den = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        if den == 0: return 0
        return abs((p2[0] - p1[0]) * (p1[1] - pt[1]) - (p1[0] - pt[0]) * (p2[1] - p1[1])) / den

    def _get_angle_between_lines(self, line1, line2):
        dx1 = line1[1][0] - line1[0][0]
        dy1 = line1[1][1] - line1[0][1]
        dx2 = line2[1][0] - line2[0][0]
        dy2 = line2[1][1] - line2[0][1]
        
        dot_product = dx1 * dx2 + dy1 * dy2
        mag1 = math.sqrt(dx1**2 + dy1**2)
        mag2 = math.sqrt(dx2**2 + dy2**2)
        
        if mag1 * mag2 == 0:
            return 0.0
            
        cos_theta = max(-1.0, min(1.0, dot_product / (mag1 * mag2)))
        angle = math.degrees(math.acos(cos_theta))
        
        if angle > 90:
            angle = 180 - angle
        return angle

    def _get_intersection_over_lowbody_area(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        
        return inter_area / float(area1) if area1 > 0 else 0.0

    def process(self, tracks, track_map, motion_mask, frame, fid, **kwargs):
        triggered = []
        curr_ids = set()
        current_time = time.time()
        
        persons = [t for t in tracks if track_map.get(int(t[4])) == ID_G_PERSON]
        low_bodies = [t for t in tracks if track_map.get(int(t[4])) == ID_PERSON_LOW]
        
        #하반신 매칭 및 발 위치 정밀 계산
        for p in persons:
            p_tid = int(p[4])
            curr_ids.add(p_tid)
            
            px1, py1, px2, py2 = p[:4]
            person_height = max(1, py2 - py1)
            p_foot = (int((px1 + px2) / 2), int(py2))
            
            best_low_track = None
            max_ioa = 0
            #해당 사람과 짝지어질 하반신을 찾습니다.
            for lb in low_bodies:
                lx1, ly1, lx2, ly2 = lb[:4]
                lcx, lcy = (lx1 + lx2) / 2, (ly1 + ly2) / 2
                
                if lcy < py1 + person_height * 0.4: 
                    continue
                    
                ioa = self._get_intersection_over_lowbody_area(lb[:4], p[:4])
                if ioa > max_ioa:
                    max_ioa = ioa
                    best_low_track = lb
                    
            curr_objects = [{'label': 'person', 'box': [int(x) for x in p[:4]], 'score': float(p[5]), 'tid': p_tid}]
            
            #하반신이 정상적으로 찾아진 경우:
            #사람 전체 박스 기준 발 위치(p_foot)와 진짜 발 위치(curr_pos)의 차이값을 lb_offsets에 저장해 둡니다. (나중에 하반신을 놓쳤을 때 쓰기 위함)
            if max_ioa >= 0.4 and best_low_track is not None:
                lx1, ly1, lx2, ly2 = best_low_track[:4]
                low_height = max(1, ly2 - ly1)
                curr_pos = (int((lx1 + lx2) / 2), int(ly2 - low_height * 0.1))
                
                self.lb_offsets[p_tid] = (curr_pos[0] - p_foot[0], curr_pos[1] - p_foot[1])
                self.lb_last_height[p_tid] = low_height
                event_bbox = tuple(best_low_track[:4])
                
                curr_objects.append({'label': 'low_body', 'box': [int(x) for x in best_low_track[:4]], 'score': float(best_low_track[5]), 'tid': int(best_low_track[4])})
            
            #컨베이어 벨트에 가려지는 등 하반신을 찾지 못한 경우:
            #과거에 저장해 두었던 오프셋(ox, oy)을 꺼내와, 대략적인 발 위치(p_foot)에 더해서 진짜 발 위치(curr_pos)를 역산해 냅니다.
            else:
                if p_tid in self.lb_offsets:
                    ox, oy = self.lb_offsets[p_tid]
                    curr_pos = (p_foot[0] + ox, p_foot[1] + oy)
                    low_height = self.lb_last_height.get(p_tid, person_height * 0.4)
                    event_bbox = (px1, py2 - low_height, px2, py2)
                else: 
                    continue
            #점프 방어
            if p_tid in self.prev:
                jump_dist = get_distance(self.prev[p_tid], curr_pos)
                if jump_dist > person_height * 0.2:
                    del self.prev[p_tid]
                    self.prev[p_tid] = curr_pos
                    continue
                
            #횡단 판별: 아직 횡단 후보자가 아닌 경우, 과거 위치와 현재 위치를 이어 선분(trajectory)을 만듭니다.
            #이 선분이 횡단선(p1, p2)과 교차(Intersect)했고, 그 진입 각도가 너무 평행하지 않다면(>= min_crossing_angle), 이 사람을 '선을 넘은 후보(candidates)'로 등록
            if p_tid in self.prev and p_tid not in self.candidates:
                trajectory = (self.prev[p_tid], curr_pos)
                for p1, p2 in self.lines:
                    if self._is_intersect(p1, p2, trajectory[0], trajectory[1]):
                        cross_angle = self._get_angle_between_lines((p1, p2), trajectory)
                        if cross_angle >= self.min_crossing_angle:
                            self.candidates[p_tid] = {
                                'person_height': person_height, 
                                'timestamp_time': current_time,
                                'line': (p1, p2), 
                                'entry_side': ccw(p1, p2, trajectory[0]), 
                                'bbox': event_bbox, 
                                'frame': frame.copy() if frame is not None else None,
                                'fid': fid,
                                'objects': curr_objects
                            }
                        break
                    
            #수직 거리 기반 최종 알람 트리거
            if p_tid in self.candidates:
                cand = self.candidates[p_tid]
                p1, p2 = cand['line']
                curr_side = ccw(p1, p2, curr_pos)
                
                #선 밖으로 진입했던 방향과 현재 방향이 반대라면
                if cand['entry_side'] != 0 and curr_side != 0 and cand['entry_side'] != curr_side:
                    perp_dist = self._get_perpendicular_distance(p1, p2, curr_pos)
                    dx = abs(p2[0] - p1[0])
                    dy = abs(p2[1] - p1[1])
                    line_tilt_angle = math.degrees(math.atan2(dy, dx))
                    
                    tilt_factor = 1.0 + (math.sin(math.radians(line_tilt_angle)) * 0.5)
                    dynamic_threshold = cand['person_height'] * self.distance_ratio * tilt_factor
                    
                    if perp_dist >= dynamic_threshold:
                        triggered.append({
                            'tid': p_tid, 
                            'bbox': cand['bbox'], 
                            'frame': cand['frame'],
                            'fid': cand['fid'],
                            'objects': cand['objects']
                        })
                        del self.candidates[p_tid]
                    else:
                        # 알람은 안 울렸지만 현재 스코어가 어디까지 가고 있는지 상시 출력
                        if p_tid in self.candidates:
                            print(f"[프레임 {fid}] ID {p_tid} 수직거리: {perp_dist:.2f} / 요구거리: {dynamic_threshold:.2f} (진행률: {(perp_dist/dynamic_threshold)*100:.1f}%)")
                        
                        
                elif current_time - cand['timestamp_time'] > self.candidate_ttl_sec: 
                    del self.candidates[p_tid]
                    
            self.prev[p_tid] = curr_pos

        for tid in list(self.prev.keys()):
            if tid not in curr_ids:
                del self.prev[tid]
                if tid in self.candidates: del self.candidates[tid]
                if tid in self.lb_offsets: del self.lb_offsets[tid]
                if tid in self.lb_last_height: del self.lb_last_height[tid]
                
        return triggered

class HelmetDetector(BaseEventDetector):
    gui_name = "NO-HELMET"
    
    def __init__(self, config, roi_poly=None, roi_lines=None):
        super().__init__(config, roi_poly, roi_lines)
        self.sessions = []
        
        self.min_streak_sec = config.get("min_streak_sec", 2.0)
        self.trigger_total_sec = config.get("trigger_total_sec", 4.0)
        self.max_gap_sec = config.get("max_gap_sec", 1.5)
        
        self.window_sec = config.get("window_sec", 30.0)
        
        self.ignore_top_ratio = config.get("ignore_top_ratio", 0.2)
        self.red_helmet_tids = set()

    def _get_roi_crop(self, frame, box):
        if frame is None:
            return None
            
        h_img, w_img = frame.shape[:2]
        x1, y1, x2, y2 = map(int, box[:4])
        
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)
        
        h_box = y2 - y1
        if h_box <= 0 or (x2 - x1) <= 0:
            return None
            
        roi_y2 = y1 + int(h_box * 0.5)
        roi = frame[y1:roi_y2, x1:x2]
        
        if roi.size == 0:
            return None
            
        return roi.copy()

    def _is_red_helmet_median(self, roi_buffer):
        if not roi_buffer:
            return False
            
        h_means, s_means, r_means = [], [], []
        
        for roi in roi_buffer:
            if roi is None or roi.size == 0:
                continue
                
            rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            r_means.append(np.mean(rgb_roi[:, :, 0]))
            h_means.append(np.mean(hsv_roi[:, :, 0]))
            s_means.append(np.mean(hsv_roi[:, :, 1]))
            
        if not h_means:
            return False
            
        med_r = np.median(r_means)
        med_h = np.median(h_means)
        med_s = np.median(s_means)
        
        return (10 <= med_h <= 40) and (med_s >= 60) and (med_r >= 100)

    def _get_intersection_over_head_area(self, head_box, person_box):
        inter_w = max(0, min(head_box[2], person_box[2]) - max(head_box[0], person_box[0]))
        inter_h = max(0, min(head_box[3], person_box[3]) - max(head_box[1], person_box[1]))
        inter_area = inter_w * inter_h
        
        head_area = max(1, (head_box[2] - head_box[0]) * (head_box[3] - head_box[1]))
        return inter_area / head_area

    def process(self, tracks, track_map, motion_mask, frame, fid, **kwargs):
        triggered = []
        helmet_tracks = kwargs.get('helmet_tracks', [])
        current_time = time.time()
        
        unhelmeted_heads = [t for t in helmet_tracks if int(t[6]) == ID_H_NO_HELMET]
        current_nh_persons = []
        
        ignore_y_thresh = 0
        if frame is not None:
            ignore_y_thresh = frame.shape[0] * self.ignore_top_ratio
        
        for p in tracks:
            p_tid = int(p[4])
            
            if p_tid in self.red_helmet_tids:
                continue
            if track_map.get(p_tid) != ID_G_PERSON: 
                continue
                
            px1, py1, px2, py2 = p[:4]
            
            if py1 <= ignore_y_thresh:
                continue
                
            if self.roi_poly is not None and self.roi_poly.size > 0:
                foot_pt = get_foot_point(*p[:4])
                if cv2.pointPolygonTest(self.roi_poly, foot_pt, False) < 0:
                    continue
                    
            person_height = max(1, py2 - py1)
            person_width = max(1, px2 - px1)

            max_ioa = 0
            nh_track_match = None
            
            for head in unhelmeted_heads:
                hx1, hy1, hx2, hy2 = head[:4]
                hcx, hcy = (hx1 + hx2) / 2, (hy1 + hy2) / 2

                if hcy > py1 + person_height * 0.4: continue
                margin = person_width * 0.15
                if hcx < px1 - margin or hcx > px2 + margin: continue

                ioa = self._get_intersection_over_head_area(head[:4], p[:4])
                if ioa > max_ioa: 
                    max_ioa = ioa
                    nh_track_match = head
                    
            if max_ioa >= 0.5 and nh_track_match is not None:
                current_nh_persons.append({
                    'tid': p_tid,
                    'head_bbox': nh_track_match[:4],
                    'person_bbox': p[:4],
                    'objects': [
                        {'label': 'person', 'box': [int(x) for x in p[:4]], 'score': float(p[5]), 'tid': p_tid},
                        {'label': 'no_helmet', 'box': [int(x) for x in nh_track_match[:4]], 'score': float(nh_track_match[5]), 'tid': int(nh_track_match[4])}
                    ]
                })
                
        for nh_p in current_nh_persons:
            matched_session = None
            for session in self.sessions:
                if session['last_tid'] == nh_p['tid'] or calculate_iou(nh_p['person_bbox'], session['last_person_bbox']) > 0.3:
                    matched_session = session
                    break
            
            roi_crop = self._get_roi_crop(frame, nh_p['head_bbox'])
                    
            if matched_session:
                gap_sec = current_time - matched_session['last_seen_time']
                if gap_sec <= self.max_gap_sec:
                    matched_session['streaks'][-1]['end_time'] = current_time
                else:
                    matched_session['streaks'].append({'start_time': current_time, 'end_time': current_time})
                
                matched_session['last_seen_time'] = current_time
                matched_session['last_tid'] = nh_p['tid']
                matched_session['last_person_bbox'] = nh_p['person_bbox']
                matched_session['bbox'] = nh_p['head_bbox']
                matched_session['frame'] = frame.copy() if frame is not None else None
                matched_session['fid'] = fid
                matched_session['objects'] = nh_p['objects']
                
                if roi_crop is not None:
                    matched_session['roi_buffer'].append(roi_crop)
            else:
                new_buffer = deque(maxlen=5)
                if roi_crop is not None:
                    new_buffer.append(roi_crop)
                    
                self.sessions.append({
                    'start_time': current_time,
                    'last_seen_time': current_time,
                    'streaks': [{'start_time': current_time, 'end_time': current_time}],
                    'last_tid': nh_p['tid'],
                    'last_person_bbox': nh_p['person_bbox'],
                    'bbox': nh_p['head_bbox'],
                    'frame': frame.copy() if frame is not None else None,
                    'fid': fid,
                    'triggered': False,
                    'roi_buffer': new_buffer,
                    'objects': nh_p['objects']
                })

        active_sessions = []
        for session in self.sessions:
            if session['last_tid'] in self.red_helmet_tids: continue
            if current_time - session['start_time'] > self.window_sec: continue
                
            total_valid_sec = 0.0
            for streak in session['streaks']:
                streak_duration = streak['end_time'] - streak['start_time']
                if streak_duration >= self.min_streak_sec:
                    total_valid_sec += streak_duration
            
            if not session['triggered'] and total_valid_sec >= self.trigger_total_sec:
                is_red_helmet = self._is_red_helmet_median(session['roi_buffer'])
                if is_red_helmet:
                    self.red_helmet_tids.add(session['last_tid'])
                else:
                    triggered.append({
                        'tid': session['last_tid'], 
                        'bbox': session['bbox'], 
                        'frame': session['frame'], 
                        'fid': session['fid'],
                        'objects': session['objects']
                    })
                session['triggered'] = True 
                
            active_sessions.append(session)
            
        self.sessions = active_sessions
        return triggered

class SignalVehicleDetector(BaseEventDetector):
    gui_name = "NO-SIGNAL"
    
    def __init__(self, config, roi_poly=None, roi_lines=None):
        super().__init__(config, roi_poly, roi_lines)
        self.history = defaultdict(lambda: deque(maxlen=30))
        self.motion_ratio = config.get("motion_threshold_ratio", 0.30)
        self.auth_grace_sec = config.get("auth_grace_sec", 120.0) 
        self.presence_threshold_sec = config.get("presence_threshold_sec", 3.0) 
        self.parked_threshold_sec = config.get("parked_threshold_sec", 60.0) 
        self.prox_ratio_x = config.get("prox_ratio_x", 1.0)
        self.prox_ratio_y = config.get("prox_ratio_y", 1.0)
        
        self.presence_start_time = {}     
        self.last_auth_time = {}          
        self.last_auth_signalman = {}     
        
        # [추가] 1프레임 미탐지 방어(Debounce)용 신호수 마지막 목격 시간 기억
        self.last_signalman_seen_time = {}

        self.stationary_anchor = {}       
        self.stationary_start_time = {}   
        
        self.last_seen_bbox = {} 
        self.last_seen_time = {} 
        self.is_parked = set() 
        
    def process(self, tracks, track_map, motion_mask, frame, fid, **kwargs):
        triggered = []
        curr_ids = set()
        current_time = time.time()
        
        if self.roi_poly.size == 0 or motion_mask is None or frame is None: return triggered
            
        h_frame, w_frame = frame.shape[:2]
        h_mask, w_mask = motion_mask.shape[:2]
        scale_x, scale_y = w_mask / float(w_frame), h_mask / float(h_frame)
        prox_x_thresh, prox_y_thresh = w_frame * self.prox_ratio_x, h_frame * self.prox_ratio_y
        
        signalman_tracks = kwargs.get('signalman_tracks', [])
        signalmen_info = [{'tid': int(t[4]), 'pt': get_foot_point(*t[:4])} for t in signalman_tracks]
        
        for t in tracks:
            if track_map.get(int(t[4])) == ID_G_TRUCK:
                curr_ids.add(int(t[4]))

        # ---------------------------------------------------------
        # [1단계] 양방향 범용 상태 상속
        # ---------------------------------------------------------
        missing_tids = [tid for tid in self.last_seen_bbox.keys() if tid not in curr_ids and (current_time - self.last_seen_time.get(tid, 0)) < 3.0]

        for curr_tid in curr_ids:
            curr_box = next((t[:4] for t in tracks if int(t[4]) == curr_tid), None)
            if curr_box is None: continue
            
            curr_fc = get_foot_point(*curr_box)
            v_size = max(curr_box[2]-curr_box[0], curr_box[3]-curr_box[1])

            for old_tid in missing_tids:
                old_box = self.last_seen_bbox[old_tid]
                iou = calculate_iou(curr_box, old_box)
                old_fc = get_foot_point(*old_box)
                dist = get_distance(curr_fc, old_fc)

                if iou > 0.3 or dist < v_size * 0.5:
                    if old_tid in self.last_auth_time: 
                        self.last_auth_time[curr_tid] = self.last_auth_time[old_tid]
                        del self.last_auth_time[old_tid]
                    if old_tid in self.last_auth_signalman: 
                        self.last_auth_signalman[curr_tid] = self.last_auth_signalman[old_tid]
                        del self.last_auth_signalman[old_tid]
                    if old_tid in self.history: 
                        self.history[curr_tid] = self.history[old_tid]
                        del self.history[old_tid]
                    if old_tid in self.presence_start_time: 
                        self.presence_start_time[curr_tid] = self.presence_start_time[old_tid]
                        del self.presence_start_time[old_tid]
                    # [추가] 목격 시간 상속
                    if old_tid in self.last_signalman_seen_time:
                        self.last_signalman_seen_time[curr_tid] = self.last_signalman_seen_time[old_tid]
                        del self.last_signalman_seen_time[old_tid]
                    if old_tid in self.stationary_anchor: 
                        self.stationary_anchor[curr_tid] = self.stationary_anchor[old_tid]
                        del self.stationary_anchor[old_tid]
                    if old_tid in self.stationary_start_time: 
                        self.stationary_start_time[curr_tid] = self.stationary_start_time[old_tid]
                        del self.stationary_start_time[old_tid]
                    if old_tid in self.is_parked: 
                        self.is_parked.add(curr_tid)
                        self.is_parked.remove(old_tid)

                    missing_tids.remove(old_tid)
                    break

        # ---------------------------------------------------------
        # [2단계] 완전 정차(Stationary) 기반 상태 업데이트 (디바운스 적용)
        # ---------------------------------------------------------
        for t in tracks:
            tid = int(t[4])
            if track_map.get(tid) != ID_G_TRUCK: continue

            x1, y1, x2, y2 = t[:4]
            fc, c_pt = get_foot_point(*t[:4]), get_center_point(*t[:4])
            v_size = max(x2 - x1, y2 - y1)

            self.last_seen_bbox[tid] = t[:4]
            self.last_seen_time[tid] = current_time

            is_in_roi = cv2.pointPolygonTest(self.roi_poly, c_pt, False) >= 0
            if is_in_roi:
                if tid not in self.stationary_anchor:
                    self.stationary_anchor[tid] = fc
                    self.stationary_start_time[tid] = current_time
                else:
                    dist_from_anchor = get_distance(self.stationary_anchor[tid], fc)
                    tolerance = max(v_size * 0.1, 15.0) 
                    
                    if dist_from_anchor > tolerance:
                        self.stationary_anchor[tid] = fc
                        self.stationary_start_time[tid] = current_time
                    else:
                        if current_time - self.stationary_start_time[tid] >= self.parked_threshold_sec:
                            self.is_parked.add(tid)
            else:
                if tid in self.stationary_anchor: del self.stationary_anchor[tid]
                if tid in self.stationary_start_time: del self.stationary_start_time[tid]
                if tid in self.is_parked: self.is_parked.remove(tid)

            if len(self.history[tid]) > 0 and get_distance(self.history[tid][-1], fc) > v_size * 0.6:
                self.history[tid].clear()
            self.history[tid].append(fc)

            has_signalman, matched_sig_tid = False, -1
            for s_info in signalmen_info:
                if abs(fc[0] - s_info['pt'][0]) <= prox_x_thresh and abs(fc[1] - s_info['pt'][1]) <= prox_y_thresh:
                    has_signalman, matched_sig_tid = True, s_info['tid']
                    break

            # [핵심 수정] 신호수 탐지 1.5초 디바운스 적용
            if has_signalman:
                self.last_signalman_seen_time[tid] = current_time
                if tid not in self.presence_start_time: self.presence_start_time[tid] = current_time
                if current_time - self.presence_start_time[tid] >= self.presence_threshold_sec:
                    self.last_auth_time[tid] = current_time
                    self.last_auth_signalman[tid] = matched_sig_tid
            else:
                last_seen = self.last_signalman_seen_time.get(tid, 0.0)
                # 1.5초 이상 신호수 트랙을 완벽히 잃어버렸을 때만 타이머 완전 초기화
                if current_time - last_seen > 1.5:
                    if tid in self.presence_start_time: del self.presence_start_time[tid]

        # ---------------------------------------------------------
        # [3단계] 이동 검지 및 알람 트리거 
        # ---------------------------------------------------------
        for t in tracks:
            tid = int(t[4])
            if track_map.get(tid) != ID_G_TRUCK: continue
            if tid not in self.is_parked: continue

            x1, y1, x2, y2 = t[:4]
            v_size = max(x2 - x1, y2 - y1)
            c_pt = get_center_point(*t[:4])
            is_in_roi = cv2.pointPolygonTest(self.roi_poly, c_pt, False) >= 0
            h_list = list(self.history[tid])

            if len(h_list) > 5:
                start_p = (sum(p[0] for p in h_list[:3])/3, sum(p[1] for p in h_list[:3])/3)
                end_p = (sum(p[0] for p in h_list[-3:])/3, sum(p[1] for p in h_list[-3:])/3)
                dist = get_distance(start_p, end_p)
                min_movement = max(v_size * 0.15, 10.0)

                if dist >= min_movement and is_in_roi:
                    mx1, my1 = max(0, int(x1 * scale_x)), max(0, int(y1 * scale_y))
                    mx2, my2 = min(w_mask, int(x2 * scale_x)), min(h_mask, int(y2 * scale_y))

                    if mx2 > mx1 and my2 > my1:
                        car_roi = motion_mask[my1:my2, mx1:mx2]
                        _, m_only = cv2.threshold(car_roi, 250, 255, cv2.THRESH_BINARY)
                        total_px = (mx2 - mx1) * (my2 - my1)

                        if total_px > 0 and (cv2.countNonZero(m_only) / total_px) > self.motion_ratio:
                            last_auth = self.last_auth_time.get(tid, 0.0)
                            time_since_auth = current_time - last_auth

                            if last_auth == 0.0 or time_since_auth > self.auth_grace_sec:
                                recent_auths = []
                                for a_tid, auth_t in self.last_auth_time.items():
                                    remain = max(0, self.auth_grace_sec - (current_time - auth_t))
                                    sig_tid = self.last_auth_signalman.get(a_tid, "Unknown")
                                    recent_auths.append({'tid': a_tid, 'remain': remain, 'auth_t': auth_t, 'sig_tid': sig_tid})

                                recent_auths.sort(key=lambda x: x['auth_t'], reverse=True)
                                triggered.append({'tid': tid, 'bbox': t[:4], 'frame': frame.copy(), 'fid': fid, 'auth_tokens': recent_auths[:1]})

                                self.history[tid].clear()
                                if tid in self.last_auth_time: del self.last_auth_time[tid]
                                if tid in self.last_auth_signalman: del self.last_auth_signalman[tid]
                                if tid in self.is_parked: self.is_parked.remove(tid)

        # ---------------------------------------------------------
        # [4단계] 상태 정리 (Cleanup)
        # ---------------------------------------------------------
        for tid in list(self.history.keys()):
            if tid not in curr_ids and (current_time - self.last_seen_time.get(tid, 0)) > 3.0: 
                del self.history[tid]

        for tid in list(self.last_auth_time.keys()):
            if current_time - self.last_auth_time[tid] > self.auth_grace_sec:
                del self.last_auth_time[tid]
                if tid in self.last_auth_signalman: del self.last_auth_signalman[tid]

        for tid in list(self.last_seen_bbox.keys()):
            if tid not in curr_ids and current_time - self.last_seen_time.get(tid, 0) > 5.0:
                del self.last_seen_bbox[tid]
                if tid in self.last_seen_time: del self.last_seen_time[tid]
                if tid in self.is_parked: self.is_parked.remove(tid) 

        for tid in list(self.presence_start_time.keys()):
            if tid not in curr_ids and (current_time - self.last_seen_time.get(tid, 0)) > 3.0: 
                del self.presence_start_time[tid]
                
        for tid in list(self.last_signalman_seen_time.keys()):
            if tid not in curr_ids and (current_time - self.last_seen_time.get(tid, 0)) > 3.0: 
                del self.last_signalman_seen_time[tid]
                
        for tid in list(self.stationary_start_time.keys()):
            if tid not in curr_ids and (current_time - self.last_seen_time.get(tid, 0)) > 3.0: 
                del self.stationary_start_time[tid]
                
        for tid in list(self.stationary_anchor.keys()):
            if tid not in curr_ids and (current_time - self.last_seen_time.get(tid, 0)) > 3.0: 
                del self.stationary_anchor[tid]

        return triggered

EVENT_REGISTRY = {
    "intrusion": IntrusionDetector, 
    "illegal_parking": ParkingDetector, 
    "conveyor_crossing": CrossingDetector, 
    "no_helmet": HelmetDetector, 
    "signal_vehicle": SignalVehicleDetector
}

# ==========================================
# [9] 터미널 마법사 및 설정 UI
# ==========================================
def capture_snapshot(url):
    """설정 마법사용 스냅샷 캡처"""
    try:
        cap = cv2.VideoCapture(sanitize_camera_url(url), cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened(): 
            return None
        ret, frame = cap.read()
        cap.release()
        return frame if ret else None
    except Exception as e:
        logger.error(f"스냅샷 캡처 실패: {e}")
        return None

def get_roi_points_scaled(frame, title, mode="poly"):
    """마우스 클릭을 통해 ROI(관심 영역)를 획득합니다."""
    pts = []
    orig_h, orig_w = frame.shape[:2]
    scale = 960 / orig_w
    disp_h = int(orig_h * scale)
    disp_frame = cv2.resize(frame, (960, disp_h))
    
    cv2.namedWindow(title)
    def mouse_cb(e, x, y, f, p):
        if e == cv2.EVENT_LBUTTONDOWN:
            if mode == "line" and len(pts) >= 2: 
                return
            pts.append([int(x / scale), int(y / scale)])
            
    cv2.setMouseCallback(title, mouse_cb)
    
    # [UX 개선] 전체 화면 적용 방법(Skip)을 명시적으로 안내
    if mode == "poly":
        logger.info(f"'{title}' 설정 - 화면을 클릭하여 점을 찍으십시오. (전체 화면 적용 시 그냥 Enter 또는 ESC)")
    else:
        logger.info(f"'{title}' 설정 - 화면을 클릭하여 선분을 그리십시오. (Enter: 완료, ESC: 취소)")
    
    while True:
        temp = disp_frame.copy()
        dp = [[int(p[0] * scale), int(p[1] * scale)] for p in pts]
        
        if mode == "line":
            if len(dp) == 1: 
                cv2.circle(temp, tuple(dp[0]), 5, (0, 0, 255), -1)
            elif len(dp) == 2: 
                cv2.line(temp, tuple(dp[0]), tuple(dp[1]), (0, 0, 255), 2)
        else:
            if len(dp) > 0: 
                cv2.polylines(temp, [np.array(dp, np.int32)], True, (0, 255, 0), 2)
                
        cv2.imshow(title, temp)
        k = cv2.waitKey(1)
        if k == 13: # Enter
            break 
        if k == 27: # ESC
            pts = []
            break 
        if mode == "line" and len(pts) == 2: 
            cv2.waitKey(500)
            break
            
    cv2.destroyWindow(title)
    return normalize_roi_points(pts, orig_w, orig_h)

def run_wizard_batch_mode(rtsp_list, existing_configs=None):
    logger.info("=== 설정 마법사 시작 ===")
    # 기존 설정을 그대로 복사하여 기반으로 삼음
    configs = existing_configs.copy() if existing_configs else {}
    
    for i in range(0, len(rtsp_list), BATCH_SIZE):
        batch = rtsp_list[i : i + BATCH_SIZE]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
            frames = list(executor.map(capture_snapshot, batch))
            
        display = []
        for idx, frm in enumerate(frames):
            if frm is None:
                blk = np.zeros((360, 640, 3), dtype=np.uint8)
                cv2.putText(blk, "Conn Fail", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                display.append(blk)
            else: 
                display.append(frm)
            
        mosaic = create_mosaic_image(display)
        cols = max(1, math.ceil(math.sqrt(len(display))))
        rows = max(1, math.ceil(len(display) / cols))
        cw = SCREEN_WIDTH // cols
        ch = SCREEN_HEIGHT // rows
        
        for idx in range(len(display)):
            r, c = divmod(idx, cols)
            cx, cy = c * cw, r * ch
            cv2.rectangle(mosaic, (cx, cy), (cx + 50, cy + 50), (255, 255, 255), -1)
            cv2.putText(mosaic, str(idx + 1), (cx + 10, cy + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
            
        cv2.imshow("Select Cameras", mosaic)
        cv2.waitKey(1)
        
        sel = input(f">> [Batch {i//BATCH_SIZE + 1}] 설정할 카메라 번호 (예: 1,3,5 / 건너뛰기: 엔터): ").strip()
        if not sel: 
            continue
        
        try:
            nums = [int(s.strip()) for s in sel.split(',')]
            for n in nums:
                if 1 <= n <= len(batch) and frames[n-1] is not None:
                    url = batch[n-1]
                    ip = extract_ip(url)
                    
                    print(f"[{ip}] 1.침입 2.주정차 3.안전모 4.횡단 5.신호수차량")
                    evts = input(f"[{ip}] 이벤트 선택 (예: 1,4): ")
                    events = []
                    
                    if '1' in evts: events.append("intrusion")
                    if '2' in evts: events.append("illegal_parking")
                    if '3' in evts: events.append("no_helmet")
                    if '4' in evts: events.append("conveyor_crossing")
                    if '5' in evts: events.append("signal_vehicle")
                    
                    roi_p = []
                    roi_l = []
                    
                    if any(e in events for e in ["intrusion", "illegal_parking", "signal_vehicle"]): 
                        roi_p = get_roi_points_scaled(frames[n-1], f"Polygon - CAM: {ip}")
                        
                    if "conveyor_crossing" in events:
                        while True:
                            l = get_roi_points_scaled(frames[n-1], f"Line - CAM: {ip}", mode="line")
                            if len(l) == 2: 
                                roi_l.extend(l)
                            if input("횡단 라인을 추가하시겠습니까? (y/n): ") != 'y': 
                                break
                                
                    configs[ip] = {
                        "url": url, 
                        "events": events, 
                        "roi_poly_norm": roi_p, 
                        "roi_lines_norm": roi_l
                    }
        except Exception as e: 
            logger.error(f"마법사 설정 중 오류 발생: {e}")
            pass
            
    cv2.destroyWindow("Select Cameras")
    return configs

# ==========================================
# [10] 카메라 제어 (FrameReader / Camera)
# ==========================================

class AnchorTrackingROIAligner:
    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        self.anchor_gray = None
        self.anchor_kp = None
        self.anchor_des = None
        self.anchor_shape = None

        self.tracking_gray = None
        self.tracking_kp = None
        self.tracking_des = None
        self.tracking_shape = None

        self.H_anchor_to_tracking = np.eye(3, dtype=np.float32)
        self.H_last_good = np.eye(3, dtype=np.float32)

        self.last_tracking_update_time = 0.0
        self.last_anchor_direct_check_time = 0.0

        self.fail_count = 0
        self.success_count = 0

        self.last_debug = {
            "status": "not_initialized",
            "method": "none",
            "raw_matches": 0,
            "good_matches": 0,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "dx": 0.0,
            "dy": 0.0,
            "angle_deg": 0.0,
            "scale": 1.0,
        }

    def _gray(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _features(self, gray):
        kp, des = self.orb.detectAndCompute(gray, None)
        return kp, des

    def set_anchor(self, frame):
        if frame is None:
            self.last_debug["status"] = "set_anchor_failed_no_frame"
            return False

        gray = self._gray(frame)
        kp, des = self._features(gray)

        if des is None or kp is None or len(kp) < MIN_GOOD_MATCHES:
            n = 0 if kp is None else len(kp)
            self.last_debug = {
                "status": f"set_anchor_failed_not_enough_features:{n}",
                "method": "anchor_init",
                "raw_matches": 0,
                "good_matches": 0,
                "inliers": 0,
                "inlier_ratio": 0.0,
                "dx": 0.0,
                "dy": 0.0,
                "angle_deg": 0.0,
                "scale": 1.0,
            }
            if DEBUG_ALIGN:
                print(f"[CCTV_Aligner] anchor 특징점 부족: {n}")
            return False

        self.anchor_gray = gray
        self.anchor_kp = kp
        self.anchor_des = des
        self.anchor_shape = frame.shape[:2]

        self.tracking_gray = gray
        self.tracking_kp = kp
        self.tracking_des = des
        self.tracking_shape = frame.shape[:2]

        self.H_anchor_to_tracking = np.eye(3, dtype=np.float32)
        self.H_last_good = np.eye(3, dtype=np.float32)

        now = time.time()
        self.last_tracking_update_time = now
        self.last_anchor_direct_check_time = now

        self.fail_count = 0
        self.success_count = 0

        self.last_debug = {
            "status": "anchor_set",
            "method": "anchor_init",
            "raw_matches": 0,
            "good_matches": len(kp),
            "inliers": 0,
            "inlier_ratio": 0.0,
            "dx": 0.0,
            "dy": 0.0,
            "angle_deg": 0.0,
            "scale": 1.0,
        }

        if DEBUG_ALIGN:
            print(f"[CCTV_Aligner] anchor 기준 프레임 등록 완료: features={len(kp)}")
        return True

    def _normalize_H(self, H):
        if H is None:
            return None
        H = H.astype(np.float32)
        if abs(float(H[2, 2])) < 1e-8:
            return None
        return H / H[2, 2]

    def _decompose_homography_rough(self, H):
        Hn = self._normalize_H(H)
        if Hn is None:
            return {"dx": 0.0, "dy": 0.0, "angle_deg": 0.0, "scale": 1.0, "perspective": 0.0}

        dx = float(Hn[0, 2])
        dy = float(Hn[1, 2])
        a = float(Hn[0, 0])
        b = float(Hn[1, 0])
        c = float(Hn[0, 1])
        d = float(Hn[1, 1])

        scale_x = (a * a + b * b) ** 0.5
        scale_y = (c * c + d * d) ** 0.5
        scale = (scale_x + scale_y) / 2.0

        angle_deg = float(np.degrees(np.arctan2(b, a)))
        perspective = max(abs(float(Hn[2, 0])), abs(float(Hn[2, 1])))

        return {"dx": dx, "dy": dy, "angle_deg": angle_deg, "scale": scale, "perspective": perspective}

    def _is_small_jitter(self, H):
        m = self._decompose_homography_rough(H)
        return (
            abs(m["dx"]) < MIN_APPLY_TRANSLATION_PX
            and abs(m["dy"]) < MIN_APPLY_TRANSLATION_PX
            and abs(m["angle_deg"]) < MIN_APPLY_ROTATION_DEG
            and abs(m["scale"] - 1.0) < MIN_APPLY_SCALE_CHANGE
            and abs(m["perspective"]) < MIN_APPLY_PERSPECTIVE
        )

    def _add_motion_debug(self, debug, H):
        m = self._decompose_homography_rough(H)
        debug["dx"] = float(m["dx"])
        debug["dy"] = float(m["dy"])
        debug["angle_deg"] = float(m["angle_deg"])
        debug["scale"] = float(m["scale"])
        debug["perspective"] = float(m["perspective"])
        return debug

    def _match_and_homography(self, src_kp, src_des, dst_kp, dst_des, dst_shape, method_name):
        if src_des is None or dst_des is None:
            return None, {"status": "descriptor_missing", "method": method_name, "raw_matches": 0, "good_matches": 0, "inliers": 0, "inlier_ratio": 0.0, "dx": 0.0, "dy": 0.0, "angle_deg": 0.0, "scale": 1.0}

        raw = self.matcher.knnMatch(src_des, dst_des, k=2)

        good = []
        for pair in raw:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)

        debug = {"status": "matching", "method": method_name, "raw_matches": len(raw), "good_matches": len(good), "inliers": 0, "inlier_ratio": 0.0, "dx": 0.0, "dy": 0.0, "angle_deg": 0.0, "scale": 1.0}

        if len(good) < MIN_GOOD_MATCHES:
            debug["status"] = f"not_enough_good_matches:{len(good)}"
            return None, debug

        src_pts = np.float32([src_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([dst_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_REPROJ_THRESH)
        if H is None or mask is None:
            debug["status"] = "homography_failed"
            return None, debug

        inliers = int(mask.sum())
        inlier_ratio = inliers / max(1, len(good))

        debug["inliers"] = inliers
        debug["inlier_ratio"] = float(inlier_ratio)
        debug = self._add_motion_debug(debug, H)

        if inliers < MIN_INLIERS:
            debug["status"] = f"not_enough_inliers:{inliers}"
            return None, debug

        if inlier_ratio < MIN_INLIER_RATIO:
            debug["status"] = f"low_inlier_ratio:{inlier_ratio:.2f}"
            return None, debug

        H = H.astype(np.float32)
        ok, reason = self._is_homography_reasonable(H, dst_shape)
        if not ok:
            debug["status"] = reason
            return None, debug

        debug["status"] = "ok"
        return H, debug

    def _is_homography_reasonable(self, H, frame_shape):
        if H is None: return False, "H_none"
        if not np.isfinite(H).all(): return False, "H_not_finite"

        h, w = frame_shape[:2]
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32).reshape(-1, 1, 2)

        try: warped = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
        except Exception: return False, "corner_transform_failed"

        if not np.isfinite(warped).all(): return False, "warped_corner_not_finite"

        orig = corners.reshape(-1, 2)
        shift = np.linalg.norm(warped - orig, axis=1)
        mean_shift = float(np.mean(shift))
        max_allowed_shift = max(w, h) * MAX_CORNER_SHIFT_RATIO

        if mean_shift > max_allowed_shift: return False, f"rejected_large_shift:{mean_shift:.1f}"

        orig_top = np.linalg.norm(orig[1] - orig[0])
        orig_bottom = np.linalg.norm(orig[2] - orig[3])
        warped_top = np.linalg.norm(warped[1] - warped[0])
        warped_bottom = np.linalg.norm(warped[2] - warped[3])

        orig_avg = max(1.0, (orig_top + orig_bottom) / 2.0)
        warped_avg = (warped_top + warped_bottom) / 2.0
        scale = warped_avg / orig_avg

        if scale < (1.0 - MAX_SCALE_CHANGE) or scale > (1.0 + MAX_SCALE_CHANGE):
            return False, f"rejected_scale:{scale:.2f}"

        if abs(float(H[2, 0])) > MAX_PERSPECTIVE_ABS or abs(float(H[2, 1])) > MAX_PERSPECTIVE_ABS:
            return False, f"rejected_perspective:{H[2,0]:.5f},{H[2,1]:.5f}"

        return True, "ok"

    def _should_update_tracking(self, debug, now):
        if debug.get("status") != "ok": return False
        if now - self.last_tracking_update_time < TRACKING_UPDATE_MIN_INTERVAL_SEC: return False
        if debug.get("inliers", 0) < TRACKING_UPDATE_MIN_INLIERS: return False
        if debug.get("inlier_ratio", 0.0) < TRACKING_UPDATE_MIN_INLIER_RATIO: return False
        return True

    def _update_tracking_reference(self, frame, kp, des, H_anchor_to_current):
        self.tracking_gray = self._gray(frame)
        self.tracking_kp = kp
        self.tracking_des = des
        self.tracking_shape = frame.shape[:2]
        self.H_anchor_to_tracking = H_anchor_to_current.astype(np.float32)
        self.last_tracking_update_time = time.time()

    def estimate_anchor_to_current(self, frame):
        if self.anchor_des is None or self.tracking_des is None:
            self.last_debug["status"] = "not_initialized"
            return np.eye(3, dtype=np.float32), False

        if frame is None:
            self.last_debug["status"] = "no_current_frame"
            return self.H_last_good.copy(), False

        gray = self._gray(frame)
        kp, des = self._features(gray)

        if kp is None or des is None or len(kp) < MIN_GOOD_MATCHES:
            self.fail_count += 1
            self.last_debug = {"status": f"current_not_enough_features:{0 if kp is None else len(kp)}", "method": "current_features", "raw_matches": 0, "good_matches": 0, "inliers": 0, "inlier_ratio": 0.0, "dx": 0.0, "dy": 0.0, "angle_deg": 0.0, "scale": 1.0}
            return self.H_last_good.copy() if KEEP_LAST_GOOD_ROI_ON_FAILURE else np.eye(3, dtype=np.float32), False

        now = time.time()

        H_tracking_to_current, dbg_tracking = self._match_and_homography(self.tracking_kp, self.tracking_des, kp, des, frame.shape[:2], method_name="tracking_to_current")

        if H_tracking_to_current is not None:
            H_anchor_to_current = H_tracking_to_current @ self.H_anchor_to_tracking
            H_anchor_to_current = H_anchor_to_current.astype(np.float32)

            ok, reason = self._is_homography_reasonable(H_anchor_to_current, frame.shape[:2])
            dbg_tracking = self._add_motion_debug(dbg_tracking, H_anchor_to_current)

            if ok:
                self.success_count += 1
                self.fail_count = 0
                self.last_debug = dbg_tracking

                if self._is_small_jitter(H_anchor_to_current):
                    self.last_debug["status"] = "skip_small_jitter_keep_identity"
                    return np.eye(3, dtype=np.float32), True

                self.H_last_good = H_anchor_to_current

                if self._should_update_tracking(dbg_tracking, now):
                    self._update_tracking_reference(frame, kp, des, H_anchor_to_current)
                    self.last_debug["status"] = "ok_tracking_updated"

                if now - self.last_anchor_direct_check_time > ANCHOR_DIRECT_CHECK_INTERVAL_SEC:
                    self._try_anchor_direct_correction(frame, kp, des)
                    self.last_anchor_direct_check_time = now

                return self.H_last_good.copy(), True
            else:
                dbg_tracking["status"] = f"anchor_to_current_rejected:{reason}"

        H_anchor_direct, dbg_anchor = self._match_and_homography(self.anchor_kp, self.anchor_des, kp, des, frame.shape[:2], method_name="anchor_to_current_fallback")

        if H_anchor_direct is not None:
            self.success_count += 1
            self.fail_count = 0
            self.last_debug = dbg_anchor
            self.last_debug = self._add_motion_debug(self.last_debug, H_anchor_direct)

            if self._is_small_jitter(H_anchor_direct):
                self.last_debug["status"] = "skip_small_jitter_anchor_fallback"
                return np.eye(3, dtype=np.float32), True

            self.H_last_good = H_anchor_direct.astype(np.float32)
            self.last_debug["status"] = "ok_anchor_fallback"

            if self._should_update_tracking(dbg_anchor, now):
                self._update_tracking_reference(frame, kp, des, self.H_last_good)
                self.last_debug["status"] = "ok_anchor_fallback_tracking_updated"

            return self.H_last_good.copy(), True

        self.fail_count += 1
        self.last_debug = dbg_tracking if dbg_tracking.get("good_matches", 0) >= dbg_anchor.get("good_matches", 0) else dbg_anchor
        self.last_debug["status"] = "failed_keep_last_good:" + str(self.last_debug.get("status"))

        if KEEP_LAST_GOOD_ROI_ON_FAILURE:
            return self.H_last_good.copy(), False
        return np.eye(3, dtype=np.float32), False

    def _try_anchor_direct_correction(self, frame, kp, des):
        H_direct, dbg = self._match_and_homography(self.anchor_kp, self.anchor_des, kp, des, frame.shape[:2], method_name="anchor_direct_drift_check")

        if H_direct is None: return False
        if dbg.get("inliers", 0) < ANCHOR_DIRECT_MIN_INLIERS: return False
        if dbg.get("inlier_ratio", 0.0) < ANCHOR_DIRECT_MIN_INLIER_RATIO: return False

        dbg = self._add_motion_debug(dbg, H_direct)

        if self._is_small_jitter(H_direct):
            self.last_debug = dbg
            self.last_debug["status"] = "anchor_direct_small_jitter_skip"
            return False

        self.H_last_good = H_direct.astype(np.float32)
        self._update_tracking_reference(frame, kp, des, self.H_last_good)
        self.last_debug = dbg
        self.last_debug["status"] = "anchor_direct_corrected_drift"
        return True

class FrameReader:
    def __init__(self, url, ip):
        self.url = sanitize_camera_url(url)
        self.ip = ip
        self.frame = None
        self.fid = 0
        self.running = True
        self.connected = False
        self.last_t = time.time()
        self.lock = threading.Lock()
        
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while self.running:
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            if not cap.isOpened(): 
                # 💡 [수정] 초기 연결 실패 로깅 (디버그 모드일때만 빈도수 조절하여 출력하도록 권장하나, 연결 실패는 중요하므로 error 처리)
                logger.error(f"🚨 [CAM:{self.ip}] RTSP 연결 실패. 5초 후 재시도합니다.")
                time.sleep(5)
                continue
                
            self.connected = True
            logger.info(f"✅ [CAM:{self.ip}] 카메라 스트림 연결 성공.")
            self.last_t = time.time()
            
            while self.running and cap.isOpened():
                if time.time() - self.last_t > WATCHDOG_TIMEOUT: 
                    # 💡 [수정] 타임아웃 로깅 레벨 격상
                    logger.error(f"🚨 [CAM:{self.ip}] 카메라 수신 타임아웃({WATCHDOG_TIMEOUT}s). 재연결을 시도합니다.")
                    break
                    
                ret, fr = cap.read()
                if not ret: 
                    logger.error(f"🚨 [CAM:{self.ip}] 프레임 읽기 실패(EOF 또는 스트림 끊김).")
                    break
                    
                if fr is not None:
                    if fr.shape[1] > 720: 
                        ratio = 720 / fr.shape[1]
                        fr = cv2.resize(fr, (720, int(fr.shape[0] * ratio)), interpolation=cv2.INTER_NEAREST)
                    with self.lock: 
                        self.frame = fr
                        self.fid += 1
                        self.last_t = time.time()
                time.sleep(0.005)
                
            self.connected = False
            try: cap.release()
            except Exception as e: logger.error(f"카메라 리소스 해제 중 예외: {e}")

    def read(self):
        with self.lock: 
            return self.frame, self.fid, self.connected

class Camera:
    def __init__(self, ip, conf, det_main, det_face, cam_id):
        self.ip = ip
        self.conf = conf
        self.cam_id = cam_id
        self.events = conf.get('events', [])
        
        self.det_main = det_main
        self.det_face = det_face
        
        self.trk_main = SimpleTracker()
        self.trk_helmet = SimpleTracker()
        self.trk_signalman = SimpleTracker()
        
        self.reader = FrameReader(conf.get('url', ''), ip)
        self.recorder = VideoRecorder(ip)
        self.motion_det = MotionDetector()
        
        self.alerted = defaultdict(set)
        self.last_evt_t = {}
        self.visual_alarms = {}

        self.fps_queue = deque(maxlen=30)
        self.current_fps = 0.0

        self.roi_poly_norm = conf.get('roi_poly_norm', [])
        self.roi_lines_norm = conf.get('roi_lines_norm', [])
        self.roi_poly = []
        self.roi_lines = []
        self.roi_frame_shape = None
        self.status_history = deque(maxlen=10)
        self._reset_alignment_state("ALIGN INIT")
        self._rebuild_handlers()

    def _reset_alignment_state(self, status_text="ALIGN RESET"):
        """ROI 자동 보정 상태를 초기화한다."""
        self.aligner = AnchorTrackingROIAligner()
        self.anchor_set = False

        self.base_roi_poly = []
        self.base_roi_lines = []
        self.aligned_roi_poly = []
        self.aligned_roi_lines = []

        self.last_align_time = 0.0
        self.align_status_text = status_text
        self.align_ok = False
        self.align_shifted = False

    def _rebuild_handlers(self):
        """현재 self.events 기준으로 이벤트 핸들러를 다시 구성한다."""
        self.handlers = {}

        for ename in self.events:
            if ename in EVENT_REGISTRY:
                self.handlers[ename] = EVENT_REGISTRY[ename](
                    SYS_CFG.get("event_config", {}).get(ename, {}),
                    self.roi_poly,
                    self.roi_lines
                )

    def update_config(self, new_conf):
        """웹 UI 등 외부에서 cameras.json이 변경되었을 때, 무중단으로 설정을 핫 리로드합니다."""
        old_events = self.events.copy()

        self.events = new_conf.get('events', [])
        self.roi_poly_norm = new_conf.get('roi_poly_norm', [])
        self.roi_lines_norm = new_conf.get('roi_lines_norm', [])

        self.roi_poly = []
        self.roi_lines = []
        self.roi_frame_shape = None

        self._reset_alignment_state("ALIGN RESET")
        self._rebuild_handlers()

        try:
            self.status_history.clear()
        except Exception:
            pass

        logger.info(
            f"🔄 [CAM:{self.ip}] 무중단 설정 리로드 완료: "
            f"{old_events} -> {self.events} | ROI aligner reset"
        )
        print(f"[CCTV_Aligner] CAM {self.cam_id} 설정 변경으로 aligner reset 완료")
        
    def _initialize_base_roi_if_needed(self, frame):
        if frame is None:
            return False

        h, w = frame.shape[:2]

        need_init = False
        if self.roi_frame_shape != frame.shape[:2]:
            need_init = True
        if self.roi_poly_norm and not self.base_roi_poly:
            need_init = True
        if self.roi_lines_norm and not self.base_roi_lines:
            need_init = True

        if not need_init:
            return True

        self.base_roi_poly = denormalize_roi_points(self.roi_poly_norm, w, h) if self.roi_poly_norm else []
        self.base_roi_lines = denormalize_roi_points(self.roi_lines_norm, w, h) if self.roi_lines_norm else []

        self.aligned_roi_poly = list(self.base_roi_poly)
        self.aligned_roi_lines = list(self.base_roi_lines)
        self.roi_frame_shape = frame.shape[:2]

        self._inject_roi_to_handlers(self.aligned_roi_poly, self.aligned_roi_lines)
        logger.info(f"[CAM:{self.cam_id}] base ROI init | poly={len(self.base_roi_poly)} lines={len(self.base_roi_lines)} shape={frame.shape[:2]}")
        return True

    def _inject_roi_to_handlers(self, roi_poly, roi_lines):
        self.roi_poly = roi_poly or []
        self.roi_lines = roi_lines or []

        for ename in self.events:
            if ename not in self.handlers: continue
            handler = self.handlers[ename]

            if self.roi_poly and len(self.roi_poly) >= 3:
                handler.roi_poly = np.array(self.roi_poly, dtype=np.int32)
            else:
                handler.roi_poly = np.empty((0, 2), dtype=np.int32)

            if hasattr(handler, "roi_lines"):
                handler.roi_lines = self.roi_lines or []

            if hasattr(handler, "lines"):
                new_lines = []
                lines = self.roi_lines or []
                for i in range(0, len(lines), 2):
                    if i + 1 < len(lines):
                        new_lines.append((lines[i], lines[i + 1]))
                handler.lines = new_lines

    def _transform_points(self, pts, H):
        if not pts: return []
        pts_np = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
        try:
            out = cv2.perspectiveTransform(pts_np, H)
        except Exception as e:
            logger.warning(f"[CAM:{self.cam_id}] ROI transform failed: {e}")
            return pts
        return out.reshape(-1, 2).astype(np.int32).tolist()

    def _update_alignment(self, frame):
        if frame is None:
            return

        # 현재 프레임 해상도 기준으로 base ROI 초기화 또는 갱신
        self._initialize_base_roi_if_needed(frame)

        # ROI가 아예 없는 경우
        if not self.base_roi_poly and not self.base_roi_lines:
            self.align_status_text = "NO ROI"
            self._inject_roi_to_handlers([], [])

            if not hasattr(self, "_no_roi_last_log_time"):
                self._no_roi_last_log_time = 0.0

            now_no_roi = time.time()
            if now_no_roi - self._no_roi_last_log_time > 60.0:
                msg = (
                    f"[CCTV_Aligner] CAM {self.cam_id} NO ROI | "
                    f"events={self.events} | ip={self.ip}"
                )
                print(msg)
                logger.warning(
                    f"[CAM:{self.cam_id}] ROI align skipped | reason=NO_ROI | "
                    f"events={self.events} | ip={self.ip}"
                )
                self._no_roi_last_log_time = now_no_roi

            return

        # 최초 anchor 등록
        if not self.anchor_set:
            ok = self.aligner.set_anchor(frame)

            if ok:
                self.anchor_set = True
                self.last_align_time = time.time()
                self.align_status_text = "ANCHOR SET"
                self.align_ok = True
                self.align_shifted = False

                msg = (
                    f"[CCTV_Aligner] CAM {self.cam_id} ANCHOR SET | "
                    f"ip={self.ip} | "
                    f"base_poly={len(self.base_roi_poly)} pts | "
                    f"base_lines={len(self.base_roi_lines)} pts"
                )
                print(msg)
                logger.info(
                    f"[CAM:{self.cam_id}] ROI anchor set | ip={self.ip} | "
                    f"base_poly={len(self.base_roi_poly)} | "
                    f"base_lines={len(self.base_roi_lines)}"
                )
            else:
                self.align_status_text = "ANCHOR FAIL"
                self.align_ok = False
                self.align_shifted = False

                dbg = getattr(self.aligner, "last_debug", {}) or {}
                status = dbg.get("status", "unknown")
                good = dbg.get("good_matches", 0)

                msg = (
                    f"[CCTV_Aligner] CAM {self.cam_id} ANCHOR FAIL | "
                    f"reason={status} | good={good} | ip={self.ip}"
                )
                print(msg)
                logger.warning(
                    f"[CAM:{self.cam_id}] ROI anchor failed | "
                    f"reason={status} | good={good} | ip={self.ip}"
                )

            return

        now = time.time()

        # edge device 부하를 줄이기 위해 지정 간격 전에는 보정 계산을 하지 않음
        if now - self.last_align_time < ALIGN_INTERVAL_SEC:
            return

        old_roi_poly = list(self.aligned_roi_poly or [])
        old_roi_lines = list(self.aligned_roi_lines or [])

        H, ok = self.aligner.estimate_anchor_to_current(frame)
        dbg = getattr(self.aligner, "last_debug", {}) or {}

        status = dbg.get("status", "unknown")
        method = dbg.get("method", "none")
        good = int(dbg.get("good_matches", 0) or 0)
        inliers = int(dbg.get("inliers", 0) or 0)
        ratio = float(dbg.get("inlier_ratio", 0.0) or 0.0)

        dx = float(dbg.get("dx", 0.0) or 0.0)
        dy = float(dbg.get("dy", 0.0) or 0.0)
        angle = float(dbg.get("angle_deg", 0.0) or 0.0)
        scale = float(dbg.get("scale", 1.0) or 1.0)
        perspective = float(dbg.get("perspective", 0.0) or 0.0)

        identity_H = np.eye(3, dtype=np.float32)
        h_shifted = not np.allclose(H, identity_H, atol=HOMOGRAPHY_IDENTITY_ATOL)

        self.align_ok = ok

        # H를 base ROI에 적용해서 이번 주기 ROI 후보 생성
        new_roi_poly = self._transform_points(self.base_roi_poly, H)
        new_roi_lines = self._transform_points(self.base_roi_lines, H)

        # ROI 변화량 계산
        def _mean_point_shift(before, after):
            if not before or not after or len(before) != len(after):
                return 0.0, 0.0

            before_np = np.array(before, dtype=np.float32)
            after_np = np.array(after, dtype=np.float32)

            if before_np.shape != after_np.shape:
                return 0.0, 0.0

            dist = np.linalg.norm(after_np - before_np, axis=1)
            return float(np.mean(dist)), float(np.max(dist))

        poly_mean_shift, poly_max_shift = _mean_point_shift(old_roi_poly, new_roi_poly)
        line_mean_shift, line_max_shift = _mean_point_shift(old_roi_lines, new_roi_lines)

        roi_mean_shift = max(poly_mean_shift, line_mean_shift)
        roi_max_shift = max(poly_max_shift, line_max_shift)

        roi_shifted = roi_max_shift >= ROI_APPLY_MIN_SHIFT_PX
        self.align_shifted = roi_shifted

        # 1) aligner가 직접 작은 jitter라고 판단한 경우
        if status == "skip_small_jitter_keep_identity":
            # 기존 aligned ROI 유지. 없으면 base ROI 주입.
            if not self.aligned_roi_poly and not self.aligned_roi_lines:
                self.aligned_roi_poly = list(self.base_roi_poly)
                self.aligned_roi_lines = list(self.base_roi_lines)

            self._inject_roi_to_handlers(self.aligned_roi_poly, self.aligned_roi_lines)

            self.align_status_text = (
                f"JITTER IGNORED {method} "
                f"status={status} g={good} i={inliers} r={ratio:.2f} "
                f"dx={dx:.1f} dy={dy:.1f} angle={angle:.2f} "
                f"scale={scale:.3f} persp={perspective:.5f} | "
                f"roi_shift max={roi_max_shift:.1f}px mean={roi_mean_shift:.1f}px | "
                f"threshold={ROI_APPLY_MIN_SHIFT_PX:.1f}px | action=roi_kept"
            )

        # 2) 매칭 성공 + 실제 ROI 좌표가 충분히 움직인 경우
        elif ok and roi_shifted:
            self.aligned_roi_poly = new_roi_poly
            self.aligned_roi_lines = new_roi_lines
            self._inject_roi_to_handlers(self.aligned_roi_poly, self.aligned_roi_lines)

            self.align_status_text = (
                f"ALIGN APPLIED {method} "
                f"status={status} g={good} i={inliers} r={ratio:.2f} "
                f"dx={dx:.1f} dy={dy:.1f} angle={angle:.2f} "
                f"scale={scale:.3f} persp={perspective:.5f} | "
                f"h_shifted={h_shifted} | "
                f"roi_shift max={roi_max_shift:.1f}px mean={roi_mean_shift:.1f}px | "
                f"poly_shift mean={poly_mean_shift:.1f}px max={poly_max_shift:.1f}px | "
                f"line_shift mean={line_mean_shift:.1f}px max={line_max_shift:.1f}px | "
                f"threshold={ROI_APPLY_MIN_SHIFT_PX:.1f}px | action=roi_updated"
            )

        # 3) 매칭은 성공했지만 실제 ROI 좌표 변화가 너무 작아서 유지
        elif ok and not roi_shifted:
            if not self.aligned_roi_poly and not self.aligned_roi_lines:
                self.aligned_roi_poly = list(self.base_roi_poly)
                self.aligned_roi_lines = list(self.base_roi_lines)

            self._inject_roi_to_handlers(self.aligned_roi_poly, self.aligned_roi_lines)

            self.align_status_text = (
                f"JITTER IGNORED {method} "
                f"status={status} g={good} i={inliers} r={ratio:.2f} "
                f"dx={dx:.1f} dy={dy:.1f} angle={angle:.2f} "
                f"scale={scale:.3f} persp={perspective:.5f} | "
                f"h_shifted={h_shifted} but roi_shift max={roi_max_shift:.1f}px "
                f"mean={roi_mean_shift:.1f}px < threshold={ROI_APPLY_MIN_SHIFT_PX:.1f}px | "
                f"action=roi_kept"
            )

        # 4) 매칭 실패 또는 homography가 위험하다고 판단된 경우
        else:
            if KEEP_LAST_GOOD_ROI_ON_FAILURE:
                if not self.aligned_roi_poly and not self.aligned_roi_lines:
                    self.aligned_roi_poly = list(self.base_roi_poly)
                    self.aligned_roi_lines = list(self.base_roi_lines)

                self._inject_roi_to_handlers(self.aligned_roi_poly, self.aligned_roi_lines)
                hold_action = "last_good_roi_kept"
            else:
                self.aligned_roi_poly = list(self.base_roi_poly)
                self.aligned_roi_lines = list(self.base_roi_lines)
                self._inject_roi_to_handlers(self.aligned_roi_poly, self.aligned_roi_lines)
                hold_action = "base_roi_restored"

            self.align_status_text = (
                f"ALIGN HOLD {method} "
                f"status={status} g={good} i={inliers} r={ratio:.2f} "
                f"dx={dx:.1f} dy={dy:.1f} angle={angle:.2f} "
                f"scale={scale:.3f} persp={perspective:.5f} | "
                f"reason={status} | action={hold_action}"
            )

        self.status_history.append(self.align_status_text)
        self.last_align_time = now

        print(f"[CCTV_Aligner] CAM {self.cam_id} {self.align_status_text}")
        logger.info(f"[CAM:{self.cam_id}] ROI align status | {self.align_status_text}")
    def process_frame(self):
        fr, fid, connected = self.reader.read()
        # [수정] 원본 영상을 바로 Recorder에 밀어넣지 않습니다. (run_logic에서 렌더링 후 삽입)
        return fr, fid, connected

    def apply_face_blur(self, frame, person_boxes):
        if frame is None or self.det_face is None: 
            return frame
            
        blur_img = frame.copy()
        
        try:
            face_conf = SYS_CFG.get("model_confidences", {}).get("FACE", 0.35)
            
            # 1. 원본 전체 프레임을 대상으로 1회만 얼굴 탐지 수행 (NPU 오버헤드 최소화 및 모델 정확도 유지)
            f_dets = self.det_face.infer(blur_img, conf_override=face_conf)
            
            for f in f_dets:
                fx1, fy1, fx2, fy2 = map(int, f[:4])
                fw, fh = fx2 - fx1, fy2 - fy1
                
                # 터무니없는 크기의 오탐 얼굴 방어 (화면의 40% 이상)
                if fw > blur_img.shape[1] * 0.4: 
                    continue 
                    
                # 2. 얼굴 BBox의 중심점 좌표 계산
                fcx = fx1 + (fw / 2.0)
                fcy = fy1 + (fh / 2.0)
                is_valid_face = False
                
                # 3. 해당 얼굴이 '사람 객체' 내부에 속하는지 검증
                for p in person_boxes:
                    px1, py1, px2, py2 = map(int, p[:4])
                    pw, ph = px2 - px1, py2 - py1
                    
                    # 얼굴이 사람 BBox 경계선이나 약간 위쪽에 걸치는 경우를 허용하기 위해 동적 여유 공간(패딩) 부여
                    pad_x = pw * 0.15          # 좌우 15% 여유
                    pad_y_top = ph * 0.25      # 머리 위쪽 25% 여유
                    pad_y_bottom = ph * 0.05   # 하단 5% 여유
                    
                    # 얼굴 중심점이 확장된 사람 ROI 내부에 포함되는지 확인
                    if (px1 - pad_x) <= fcx <= (px2 + pad_x) and (py1 - pad_y_top) <= fcy <= (py2 + pad_y_bottom):
                        is_valid_face = True
                        break
                        
                # 4. 검증을 통과한 유효 얼굴(사람 내부)에만 모자이크 렌더링
                if is_valid_face:
                    roi = blur_img[fy1:fy2, fx1:fx2]
                    if roi.size > 0:
                        small = cv2.resize(roi, (max(1, fw//15), max(1, fh//15)), interpolation=cv2.INTER_LINEAR)
                        blur_img[fy1:fy2, fx1:fx2] = cv2.resize(small, (fw, fh), interpolation=cv2.INTER_NEAREST)
                        
        except Exception as e: 
            logger.error(f"모자이크 처리 실패: {e}")
            
        return blur_img

    def run_logic(self, fr, fid, d_main_res, d_helmet_res, t_signalman):
        if fr is None:
            return [], [], [], {}, []

        now_t = time.time()
        self.fps_queue.append(now_t)
        if len(self.fps_queue) > 1:
            time_diff = self.fps_queue[-1] - self.fps_queue[0]
            self.current_fps = len(self.fps_queue) / time_diff if time_diff > 0 else 0.0

        self._update_alignment(fr)
        motion_mask = self.motion_det.apply(fr)

        d_main_filtered = [d for d in d_main_res if int(d[5]) not in [ID_H_HELMET, ID_H_NO_HELMET]]
        t_main = self.trk_main.update(d_main_filtered)
        
        # d_helmet_filtered = [d for d in d_helmet_res if int(d[5]) == ID_H_NO_HELMET]
        # t_helmet = self.trk_helmet.update(d_helmet_filtered)
        t_helmet = self.trk_helmet.update(d_helmet_res)

        now = time.time()
        current_alarms = {} 
        track_map_main = {int(t[4]): int(t[6]) for t in t_main}
        score_map_main = {int(t[4]): round(float(t[5]), 2) for t in t_main}
        newly_triggered_events = []

        record_fr = fr.copy()

        for ename, handler in self.handlers.items():
            # [수정] 신호수 트랙 데이터를 kwargs로 묶어 핸들러로 전달
            kwargs = {}
            if ename == "no_helmet": kwargs['helmet_tracks'] = t_helmet
            if ename == "signal_vehicle": kwargs['signalman_tracks'] = t_signalman
            
            try:
                triggered = handler.process(t_main, track_map_main, motion_mask, fr, fid, **kwargs)
            except Exception as e:
                logger.error(f"🚨 [CAM:{self.ip}] {ename} 핸들러 처리 중 예외 발생: {e}\n{traceback.format_exc()}")
                continue
            
            for ev in triggered:
                tid = ev['tid']
                bbox = ev['bbox']
                ev_frame = ev.get('frame') if ev.get('frame') is not None else fr
                cooldown = SYS_CFG.get("event_config", {}).get(ename, {}).get("cooldown_sec", 600)
                
                actual_score = score_map_main.get(tid, 0.95)
                objects_meta = ev.get('objects', [{'label': ename, 'box': [int(x) for x in bbox], 'score': actual_score, 'tid': tid}])
                
                if ename not in self.alerted[tid] and (now - self.last_evt_t.get(ename, 0) >= cooldown):
                    objs_log_str = " | ".join([f"{o['label']}({o['score']:.2f}): {o['box']}" for o in objects_meta])
                    
                    log_msg = (
                        f"🔥 [EVENT TRIGGERED] CAM:{self.cam_id}({self.ip}) | Event:{ename} | "
                        f"TermID:{SYS_CFG.get('terminal_id', '99999')} | TID:{tid} | FPS:{self.current_fps:.1f} | "
                        f"Objects -> {objs_log_str}"
                    )
                    logger.warning(log_msg)
                    
                    blur_face_option = SYS_CFG.get("event_config", {}).get(ename, {}).get("blur_face", True)
                    person_boxes = [t for t in t_main if int(t[6]) in [ID_G_PERSON, ID_PERSON_LOW, ID_REFLECTIVE_VEST]]
                    saved_img = self.apply_face_blur(ev_frame, person_boxes) if blur_face_option else ev_frame
                    
                    event_trajectories = {}
                    for obj in objects_meta:
                        obj_tid = obj.get('tid')
                        if obj_tid in self.trk_main.tracks:
                            event_trajectories[obj_tid] = list(self.trk_main.tracks[obj_tid]['history'])
                        elif obj_tid in self.trk_helmet.tracks:
                            event_trajectories[obj_tid] = list(self.trk_helmet.tracks[obj_tid]['history'])
                    
                    auth_tokens = ev.get('auth_tokens', None)

                    save_event_image_with_mark(
                        frame=saved_img, ip=self.ip, event_type=ename, bbox=bbox, tid=tid, 
                        terminal_id=SYS_CFG.get("terminal_id", "99999"), cctv_id=self.cam_id, 
                        objects_meta=objects_meta, trajectories=event_trajectories,
                        auth_tokens=auth_tokens 
                    )
                    
                    self.recorder.trigger(ename, objects_meta=objects_meta) 
                    self.alerted[tid].add(ename)
                    self.last_evt_t[ename] = now
                    
                    newly_triggered_events.append({'event_name': ename, 'objects': objects_meta})
                    
                current_alarms[tid] = ename
        
        alarm_duration = SYS_CFG.get("VISUAL_ALARM_DURATION", 5.0)
        for tid, ename in current_alarms.items(): 
            self.visual_alarms[tid] = {'evt': ename, 'expire': now + alarm_duration}
            
        for tid in list(self.visual_alarms.keys()):
            if now > self.visual_alarms[tid]['expire']: 
                del self.visual_alarms[tid]
                
        if record_fr is not None:
            for t in t_main:
                t_id = int(t[4])
                is_alarmed = t_id in current_alarms
                
                color = (0, 0, 255) if is_alarmed else (0, 255, 0)
                thickness = 3 if is_alarmed else 1
                bx1, by1, bx2, by2 = map(int, t[:4])
                
                cv2.rectangle(record_fr, (bx1, by1), (bx2, by2), color, thickness)
                
                if t_id in self.trk_main.tracks:
                    hist = list(self.trk_main.tracks[t_id]['history'])
                    if len(hist) > 1:
                        cv2.polylines(record_fr, [np.array(hist, np.int32)], False, color, thickness, cv2.LINE_AA)
                        
        # [수정] draw 메서드에서 렌더링할 수 있도록 t_signalman을 리턴에 포함
        return t_main, t_helmet, t_signalman, {t: info['evt'] for t, info in self.visual_alarms.items()}, newly_triggered_events

    def draw(self, fr, t_main, t_helmet, t_signalman, alarms, connected=True):
        if fr is None or not connected:
            blank = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(blank, f"CAM {self.cam_id} NO SIGNAL", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.putText(blank, self.ip, (50, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            return blank

        h_frame, w_frame = fr.shape[:2]
        
        if len(alarms) > 0: 
            cv2.rectangle(fr, (0, 0), (w_frame, h_frame), (0, 0, 255), 20)
            
        if len(self.roi_poly) > 2: 
            cv2.polylines(fr, [np.array(self.roi_poly, np.int32)], True, (0, 255, 255), 2)
        if self.roi_lines:
            for i in range(0, len(self.roi_lines), 2):
                if i + 1 < len(self.roi_lines): 
                    cv2.line(fr, tuple(self.roi_lines[i]), tuple(self.roi_lines[i+1]), (0, 0, 255), 2)

        # -----------------------------------------------------------
        # [핵심] 카메라별 설정된 이벤트에 따라 화면에 그릴 클래스(ID) 동적 필터링
        # -----------------------------------------------------------
        allowed_classes = set()
        if "signal_vehicle" in self.events:
            allowed_classes.add(ID_G_TRUCK) 
        if "no_helmet" in self.events or "conveyor_crossing" in self.events or "intrusion" in self.events:
            allowed_classes.update([ID_G_PERSON, ID_PERSON_LOW]) 
        if "illegal_parking" in self.events or "intrusion" in self.events:
            allowed_classes.update(TARGET_VEHICLES) 

        # 1. Main Tracker BBox 렌더링
        for t in t_main:
            tid = int(t[4])
            cls_id = int(t[6])
            is_alarmed = tid in alarms
            
            # 알람이 울린 객체가 아니고, 해당 카메라의 감시 대상 클래스가 아니면 화면에서 깔끔하게 숨김
            if not is_alarmed and cls_id not in allowed_classes: 
                continue 

            color = (0, 0, 255) if is_alarmed else (0, 255, 0)
            thickness = 2 if is_alarmed else 1
            
            if tid in self.trk_main.tracks:
                hist = list(self.trk_main.tracks[tid]['history'])
                if len(hist) > 1:
                    cv2.polylines(fr, [np.array(hist, np.int32)], False, color, 1, cv2.LINE_AA)

            if cls_id == ID_G_PERSON: label = f"Person [{tid}]"
            elif cls_id == ID_PERSON_LOW: label, color = f"LowBody [{tid}]", (0, 150, 0)
            elif cls_id in TARGET_VEHICLES: label, color = f"Vehicle [{tid}]", (255, 100, 0)
            else: label = f"OBJ [{tid}]"

            if is_alarmed: 
                color = (0, 0, 255)
                label = f"ALARM: {label}"
                
            cv2.rectangle(fr, (int(t[0]), int(t[1])), (int(t[2]), int(t[3])), color, thickness)
            cv2.putText(fr, label, (int(t[0]), int(t[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 2. Signalman 커스텀 모델 전용 Tracker BBox 렌더링
        if "signal_vehicle" in self.events:
            for t in t_signalman:
                tid = int(t[4])
                color, thickness = (0, 255, 255), 2 
                cv2.rectangle(fr, (int(t[0]), int(t[1])), (int(t[2]), int(t[3])), color, thickness)
                cv2.putText(fr, f"Signalman [{tid}]", (int(t[0]), int(t[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
        # 3. Helmet Tracker BBox 렌더링 (정상 헬멧과 미착용 분리)
        if "no_helmet" in self.events:
            for t in t_helmet:
                tid = int(t[4])
                cls_id = int(t[6]) # 0: Helmet, 1: No-Helmet
                
                # [수정] 헬멧 착용 여부에 따라 라벨과 색상을 명확히 분리
                if cls_id == ID_H_HELMET:
                    color = (0, 255, 0) # 초록색
                    label = f"Helmet [{tid}]"
                    thickness = 2
                else:
                    color = (0, 0, 255) # 빨간색
                    label = f"Head [{tid}]"
                    thickness = 3 if tid in alarms else 2
                    
                cv2.rectangle(fr, (int(t[0]), int(t[1])), (int(t[2]), int(t[3])), color, thickness)
                cv2.putText(fr, label, (int(t[0]), int(t[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 좌측 상단 카메라 ID 및 FPS
        cv2.rectangle(fr, (0, 0), (100, 40), (0, 0, 0), -1) 
        cv2.putText(fr, f"CAM {self.cam_id}", (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        fps_color = (0, 255, 0) if self.current_fps >= 10.0 else (0, 0, 255)
        cv2.putText(fr, f"FPS: {self.current_fps:.1f}", (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.35, fps_color, 1)
        
        active_alarms = set(alarms.values())
        
        # 우측 상단 이벤트 메뉴
        menu_height = len(self.events) * 20 + 10
        overlay = fr.copy()
        cv2.rectangle(overlay, (w_frame - 150, 0), (w_frame, menu_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, fr, 0.5, 0, fr)
        
        y_pos = 15
        for evt in self.events:
            display_name = EVENT_REGISTRY[evt].gui_name if evt in EVENT_REGISTRY else evt.upper()
            color = (0, 0, 255) if evt in active_alarms else (0, 255, 0)
            prefix = "[!] " if evt in active_alarms else " -  "
            
            cv2.putText(fr, f"{prefix}{display_name}", (w_frame - 145, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
            y_pos += 20
            
        # -----------------------------------------------------------
        # Signalman Auth 상태창 렌더링
        # -----------------------------------------------------------
        if "signal_vehicle" in self.events and "signal_vehicle" in self.handlers:
            sv_handler = self.handlers["signal_vehicle"]
            current_time = time.time()
            display_items = []

            for a_tid, auth_t in list(sv_handler.last_auth_time.items()):
                remain = sv_handler.auth_grace_sec - (current_time - auth_t)
                if remain > 0:
                    auth_time_str = datetime.datetime.fromtimestamp(auth_t).strftime('%H:%M:%S')
                    is_visible = any(int(t[4]) == a_tid for t in t_main if int(t[6]) == ID_G_TRUCK)
                    status_text = "Tracking" if is_visible else "Hidden"
                    sig_id = sv_handler.last_auth_signalman.get(a_tid, "Unknown")
                    
                    sort_score = auth_t
                    if a_tid in alarms: sort_score = float('inf')
                    
                    display_items.append({
                        'tid': a_tid, 'sort_val': sort_score,
                        'line1': f"Auth: {auth_time_str} | Remain: {remain:.1f}s ({status_text})",
                        'line2': f"Auth by: Signalman [{sig_id}]",
                        'color': (0, 255, 0) if is_visible else (0, 180, 0)
                    })

            current_trucks = [int(t[4]) for t in t_main if int(t[6]) == ID_G_TRUCK]
            auth_tids = [item['tid'] for item in display_items]

            for t_tid in current_trucks:
                if t_tid not in auth_tids:
                    is_alarming = t_tid in alarms
                    base_sort = float('inf') if is_alarming else 0
                    
                    if is_alarming:
                        display_items.append({
                            'tid': t_tid, 'sort_val': base_sort + 3,
                            'line1': "Status: UNAUTH (ALARM)",
                            'line2': "Reason: Moving without Signalman",
                            'color': (0, 0, 255)
                        })
                    elif t_tid in sv_handler.presence_start_time:
                        wait_sec = current_time - sv_handler.presence_start_time[t_tid]
                        display_items.append({
                            'tid': t_tid, 'sort_val': base_sort + 2,
                            'line1': f"Wait: {wait_sec:.1f}s / {sv_handler.presence_threshold_sec}s",
                            'line2': "Authenticating Signalman...",
                            'color': (0, 165, 255) 
                        })
                    elif t_tid in sv_handler.is_parked:
                        display_items.append({
                            'tid': t_tid, 'sort_val': base_sort + 1,
                            'line1': "Status: PARKED (Monitoring)",
                            'line2': "Awaiting Signalman",
                            'color': (255, 150, 0) 
                        })
                    else:
                        dwell_sec = current_time - sv_handler.stationary_start_time.get(t_tid, current_time)
                        display_items.append({
                            'tid': t_tid, 'sort_val': -1, 
                            'line1': f"Status: ARRIVING (Stop: {dwell_sec:.0f}s / {sv_handler.parked_threshold_sec}s)",
                            'line2': "Ignoring Move (Parking in progress)",
                            'color': (180, 180, 180) 
                        })

            display_items.sort(key=lambda x: x['sort_val'], reverse=True)
            display_items = display_items[:1]

            box_w, box_h = 340, 35 + max(1, len(display_items)) * 40
            x_start, y_start = w_frame - box_w - 20, h_frame - box_h - 20

            overlay2 = fr.copy()
            cv2.rectangle(overlay2, (x_start, y_start), (x_start + box_w, y_start + box_h), (0, 0, 0), -1)
            cv2.addWeighted(overlay2, 0.6, fr, 0.4, 0, fr)
            cv2.putText(fr, "Signalman Auth", (x_start + 10, y_start + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            if not display_items:
                cv2.putText(fr, "No active tokens", (x_start + 10, y_start + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            else:
                for i, item in enumerate(display_items):
                    cv2.putText(fr, item['line1'], (x_start + 10, y_start + 45 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, item['color'], 1)
                    cv2.putText(fr, item['line2'], (x_start + 10, y_start + 65 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, item['color'], 1)
                    
        return fr
# ==========================================
# [11]  Platform 송수신 모듈
# ==========================================
def get_system_temperature():
    """OS 환경(Linux, Edge Device 등)에 맞게 시스템 온도를 안전하게 수집합니다."""
    try:
        # 1차 시도: psutil을 통한 센서 온도 읽기
        if hasattr(psutil, "sensors_temperatures"):
            temps = psutil.sensors_temperatures()
            if temps:
                for name, entries in temps.items():
                    for entry in entries:
                        return float(entry.current)
                        
        # 2차 시도: 리눅스/엣지 단말(Jetson, Raspberry Pi 등)의 하드웨어 파일 직접 참조
        temp_path = "/sys/class/thermal/thermal_zone0/temp"
        if os.path.exists(temp_path):
            with open(temp_path, "r") as f:
                return float(f.read().strip()) / 1000.0
    except Exception as e:
        logger.debug(f"온도 센서 읽기 실패 (해당 OS 미지원): {e}")
        
    return 0.0 # 센서가 없는 PC 환경 등의 폴백(Fallback)

class HealthCheckDaemon:
    def __init__(self, terminal_id, version="v1.1.0", interval_sec=60):
        self.terminal_id = terminal_id
        self.version = version
        self.interval = interval_sec
        self.running = True
        self.url = "https://tmlsafety.hudaters.net/receiver/api/v1/cctv/health"
        
        # 데몬 스레드로 실행하여 메인 프로세스 종료 시 강제 종료되도록 허용
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        logger.info(f"🩺 [Health Check] 백그라운드 헬스 체크 데몬 시작 (주기: {self.interval}초)")

    def _run(self):
        while self.running:
            try:
                # 자원 수집
                cpu = psutil.cpu_percent(interval=1.0)
                mem = psutil.virtual_memory().percent
                temp = get_system_temperature()
                
                # ISO 8601 포맷
                kst = pytz.timezone('Asia/Seoul')
                reported_at = datetime.datetime.now(kst).strftime('%Y-%m-%dT%H:%M:%S')

                data = {
                    "terminalId": str(self.terminal_id),
                    "reportedAt": reported_at,
                    "cpuUsage": round(cpu, 1),
                    "memoryUsage": round(mem, 1),
                    "temperature": round(temp, 1),
                    "softwareVersion": self.version
                }

                # requests 모듈은 딕셔너리를 data= 에 넘기면 자동으로 application/x-www-form-urlencoded 로 처리합니다.
                headers = {"accept": "application/json"}
                
                response = requests.post(self.url, headers=headers, data=data, timeout=10, verify=False)
                
                if response.status_code == 200:
                    logger.debug(f"🩺 [Health Check] 전송 성공 (CPU: {data['cpuUsage']}%, Mem: {data['memoryUsage']}%)")
                else:
                    logger.error(f"⚠️ [Health Check] API 응답 에러 (상태코드: {response.status_code}) - {response.text}")
                    
            except Exception as e:
                logger.error(f"⚠️ [Health Check] 네트워크 연결 예외 발생: {e}")
            
            # interval(300초)을 통으로 sleep하지 않고, 1초마다 running 상태를 체크하여 빠른 셧다운을 지원
            for _ in range(self.interval):
                if not self.running:
                    break
                time.sleep(1)

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)

def main():
    # 1. argparse를 활용한 실행 옵션 분기 (기본값: CLI 모드)
    parser = argparse.ArgumentParser(description="Raspberry Pi Edge AI CCTV Event Detection")
    parser.add_argument('--gui', action='store_true', help="GUI 모드를 활성화하여 모니터에 영상을 렌더링합니다.")
    args = parser.parse_args()
    
    is_gui_mode = args.gui

    if not is_gui_mode:
        logger.info("[시스템 모드] CLI (Headless) 모드로 동작합니다. (렌더링 생략으로 CPU 부하 최소화)")
    else:
        logger.info("[시스템 모드] GUI 모드로 동작합니다. (--gui 플래그 활성화됨)")
        
    global DEBUG_MODE
    logger.info("[System] 단일 스크립트 기반 YOLOv8 모듈화 시스템 초기화 완료")
    
    rtsp_list = load_rtsp_list_from_csv(CAMERA_LIST_FILE)
    if not rtsp_list:
        logger.error(f"카메라 목록 파일({CAMERA_LIST_FILE})을 확인하십시오.")
        return
        
    config_file = os.path.join(PROJECT_ROOT, "cameras.json")
    camera_configs = {}
    
    debug_ans = input(">> 디버그 모드를 활성화하시겠습니까? (상세 로그 출력) [y/N]: ").strip().lower()
    DEBUG_MODE = True if debug_ans == 'y' else False
    if DEBUG_MODE:
        _log_level_str = SYS_CFG.get("logging", {}).get("level", "INFO").upper()
        logger.setLevel(getattr(logging, _log_level_str, logging.INFO))
        logger.debug("🛠️ 디버그 모드가 활성화되었습니다. 상세 로깅이 시작됩니다.")
    
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r', encoding='utf-8') as f: 
                camera_configs = json.load(f)
        except Exception as e: 
            logger.error(f"cameras.json 로드 실패: {e}")
            pass
            
        reset_ans = input(">> 기존 설정(cameras.json)을 무시하고 ROI 및 이벤트를 재설정하시겠습니까? [y/N]: ").strip().lower()
        if reset_ans == 'y':
            logger.info("기존 설정을 무시하고 터미널 마법사를 실행합니다.")
            camera_configs = run_wizard_batch_mode(rtsp_list, camera_configs)
            try:
                with open(config_file, 'w', encoding='utf-8') as f: 
                    json.dump(camera_configs, f, indent=4)
            except: pass
    else:
        logger.warning("설정 파일(cameras.json)이 없어 터미널 마법사를 실행합니다.")
        camera_configs = run_wizard_batch_mode(rtsp_list, {})
        try:
            with open(config_file, 'w', encoding='utf-8') as f: 
                json.dump(camera_configs, f, indent=4)
        except: pass

    try:
        logger.info("🔥 [최적화] DeepX 단일 통합 모델을 VPU에 할당 중...")
        # [핵심] 여러 모델을 로드하지 않고, 모든 클래스가 있는 signalman.dxnn 하나만 로드합니다.
        d_main = YoLoDeepX(os.path.join(PROJECT_ROOT, "signalman.dxnn"))
        # 얼굴 모자이크용은 독립적으로 유지
        d_face = YoLoDeepX(SYS_CFG["models"]["FACE"]) 
    except Exception as e:
        logger.error(f"모델 로드 실패. 경로를 확인하십시오: {e}")
        return

    cams = []
    for i, rtsp in enumerate(rtsp_list):
        ip = extract_ip(rtsp)
        conf = camera_configs.get(ip)
        
        if not conf or not conf.get('events'): continue
        conf['url'] = rtsp
        
        # [수정] 헬멧, 신호수 파라미터 싹 지우고 통합 모델(d_main) 하나만 넘김
        cams.append(Camera(ip, conf, d_main, d_face, cam_id=i+1))
        logger.info(f"Loaded [CAM {i+1}]: {ip}")

    # 환경 변수 스로틀링 기준
    target_fps = SYS_CFG.get("REC_FPS", 15)
    main_conf = SYS_CFG["model_confidences"]["MAIN"]
    helmet_conf = SYS_CFG["model_confidences"]["HELMET"]
    person_conf = SYS_CFG.get("model_confidences", {}).get("PERSON", 0.35)  # [추가] 설정값 로드
    loop_count = 0
    fps_calc_interval = 30
    last_fps_time = time.time()
    cpu_usage = 0.0
    dynamic_delay = 1.0 / target_fps
    
    terminal_id = SYS_CFG.get("terminal_id", "99999")
    software_version = "v1.1.0"  
    health_daemon = HealthCheckDaemon(terminal_id=terminal_id, version=software_version, interval_sec=60)
    
    last_config_mtime = 0
    if os.path.exists(config_file):
        last_config_mtime = os.path.getmtime(config_file)
        
    RAM_DISK_DIR = "/dev/shm/cctv_frames"
    if not os.path.exists(RAM_DISK_DIR):
        try: os.makedirs(RAM_DISK_DIR, exist_ok=True)
        except: RAM_DISK_DIR = "./web_frames" 

    # [수정] 카메라별 5초 지연 큐(Debounce Queue) 및 타이머 초기화
    event_save_queues = {c.ip: [] for c in cams}
    last_event_times = {c.ip: 0.0 for c in cams}

    try:
        psutil.cpu_percent(interval=None)
        
        while True:
            start_time = time.time()
            
            if loop_count > 0 and loop_count % 45 == 0 and os.path.exists(config_file):
                current_mtime = os.path.getmtime(config_file)
                if current_mtime > last_config_mtime:
                    logger.info("🛠️ [System] cameras.json 변경 감지. 카메라 설정을 무중단 핫 리로드합니다.")
                    try:
                        with open(config_file, 'r', encoding='utf-8') as f:
                            new_configs = json.load(f)
                        for c in cams:
                            if c.ip in new_configs:
                                c.update_config(new_configs[c.ip])
                        last_config_mtime = current_mtime
                        
                        # [추가] 만약 system_config.json 도 함께 체크하거나 리로드 구조가 있다면 
                        # 여기에서 person_conf = SYS_CFG.get("model_confidences", {}).get("PERSON", 0.35) 를 갱신할 수 있습니다.
                    except Exception as e:
                        logger.error(f"핫 리로드 중 예외 발생: {e}")

            loop_count += 1
            
            if loop_count % fps_calc_interval == 0:
                current_time = time.time()
                elapsed_time = current_time - last_fps_time
                actual_fps = fps_calc_interval / elapsed_time
                
                cpu_usage = psutil.cpu_percent(interval=None)
                
                if cpu_usage > 85: 
                    target_fps = max(5, target_fps - 2)
                elif cpu_usage < 60: 
                    target_fps = min(15, target_fps + 1)
                    
                dynamic_delay = 1.0 / target_fps
                
                if DEBUG_MODE:
                    logger.debug(f"⏱️ [Performance Debug] CPU: {cpu_usage:.1f}% | 실제 속도: {actual_fps:.1f} FPS (목표: {target_fps} FPS)")
                
                last_fps_time = current_time

            if loop_count % 300 == 0: 
                gc.collect()
                mem_usage = psutil.virtual_memory().percent
                q_size = IMAGE_SAVER_POOL._work_queue.qsize() if hasattr(IMAGE_SAVER_POOL, '_work_queue') else 0
                
                if mem_usage > 80 or q_size > 20:
                    logger.warning(f"⚠️ [System Health] CPU: {cpu_usage:.1f}% | Mem: {mem_usage:.1f}% | API Queue: {q_size}")
            
            raw_data = [c.process_frame() for c in cams]
            final_imgs = []
            
            for idx, res in enumerate(raw_data):
                fr, fid, connected = res
                
                if connected and fr is not None and loop_count % 100 == 0:
                    try:
                        small_fr = cv2.resize(fr, (640, 360))
                        save_path = os.path.join(RAM_DISK_DIR, f"{cams[idx].ip}.jpg")
                        cv2.imwrite(save_path, small_fr, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    except Exception as e:
                        pass
                
                if not cams[idx].events:
                    if connected and fr is not None:
                        final_imgs.append(cams[idx].draw(fr, [], [], {}, True))
                    else:
                        final_imgs.append(cams[idx].draw(None, [], [], {}, False))
                    continue
                
                if not connected:
                    final_imgs.append(cams[idx].draw(None, [], [], {}, False))
                    continue
                
                # ---------------------------------------------------------
                # [수정] 사람(2) 및 신호수(5) 클래스 전용 Confidence 개별 적용
                # ---------------------------------------------------------
                base_conf = min(main_conf, person_conf, helmet_conf)
                raw_dets = cams[idx].det_main.infer(fr, conf_override=base_conf)
                
                d_main_res_list = []
                d_helmet_res = []
                d_signal_res = []
                
                # 파이썬 리스트 컴프리헨션 수준에서 클래스별로 결과를 쪼개서 분배합니다. (NPU 병목 해소)
                for d in raw_dets:
                    cls_id = int(d[5])
                    conf = float(d[4])
                    
                    if cls_id == ID_REFLECTIVE_VEST: # 신호수(5)
                        if conf >= person_conf:
                            d_main_res_list.append(d) 
                            d_signal_res.append(d)
                    elif cls_id in [ID_H_HELMET, ID_H_NO_HELMET]: # 헬멧(0, 1)
                        if conf >= helmet_conf:
                            d_helmet_res.append(d)
                    elif cls_id in [ID_G_PERSON, ID_PERSON_LOW]: # 사람(2, 4)
                        if conf >= person_conf:
                            d_main_res_list.append(d)
                    else: # 차량 등(3, 6)
                        if conf >= main_conf:
                            d_main_res_list.append(d)
                            
                t_main_input = np.array(d_main_res_list) if len(d_main_res_list) > 0 else np.empty((0, 6))
                
                # 쪼개진 데이터를 각각의 트래커에 던져줍니다.
                t_signalman = cams[idx].trk_signalman.update(d_signal_res)
                
                t_main, t_helmet, t_signalman, alarms, new_events = cams[idx].run_logic(
                    fr, fid, t_main_input, d_helmet_res, t_signalman
                )
                
                # -----------------------------------------------------------
                # BBox와 알람이 그려진 프레임을 생성하여 녹화기에 주입
                # -----------------------------------------------------------
                if connected and fr is not None:
                    recorded_fr = fr.copy()
                    recorded_fr = cams[idx].draw(recorded_fr, t_main, t_helmet, t_signalman, alarms, True)
                    
                    cams[idx].recorder.update(recorded_fr)
                    
                    if is_gui_mode:
                        final_imgs.append(recorded_fr)
                else:
                    if is_gui_mode:
                        final_imgs.append(cams[idx].draw(None, [], [], [], {}, False))
                # -----------------------------------------------------------
                    
                if new_events:
                    # [수정] 디스크에 바로 쓰지 않고, 큐에 스택(Stacking)하며 이벤트 시간 갱신
                    # 메인 루프 참조 문제 방지를 위해 fr.copy() 사용
                    event_save_queues[cams[idx].ip].append((fid, fr.copy()))
                    last_event_times[cams[idx].ip] = time.time()
                    
                    for ev_data in new_events:
                        api_payload = []
                        for obj in ev_data['objects']:
                            api_payload.append({
                                "box": obj['box'],
                                "label": obj['label'],
                                "score": obj['score']
                            })
                        logger.info(f"[{cams[idx].ip}] 알람 API 페이로드 덤프 ({ev_data['event_name']}): {json.dumps(api_payload)}")

            # [수정] 5초 지연(Debounce) 만료 체크 및 큐 비우기 (Flush)
            now_time = time.time()
            for c in cams:
                ip = c.ip
                q = event_save_queues.get(ip, [])
                
                # 큐에 데이터가 있고, 마지막 이벤트로부터 5초 이상 경과했다면
                if len(q) > 0 and (now_time - last_event_times.get(ip, 0.0) > 5.0):
                    logger.debug(f"[{ip}] 5초 대기 완료. 쌓인 큐({len(q)}장)를 비동기 저장합니다.")
                    for item_fid, item_fr in q:
                        event_img_path = os.path.join(EVENT_ROOT_DIR, f"cam_{ip}_{item_fid}.jpg")
                        # 디스크 쓰기 병목 방지를 위해 스레드 풀에 작업 위임 (Off-loading)
                        IMAGE_SAVER_POOL.submit(cv2.imwrite, event_img_path, item_fr)
                    q.clear()

            if is_gui_mode:
                if final_imgs:
                    cv2.imshow("Monitor", create_mosaic_image(final_imgs))
                if cv2.waitKey(1) == ord('q'): 
                    break

            sleep_time = dynamic_delay - (time.time() - start_time)
            if sleep_time > 0: 
                time.sleep(sleep_time)

    except KeyboardInterrupt: 
        logger.info("[종료] 사용자에 의해 시스템이 중단되었습니다.")
    except Exception as e:
        logger.error(f"[치명적 오류] {e}\n{traceback.format_exc()}")
    finally:
        if 'health_daemon' in locals():
            logger.info("🩺 [Health Check] 데몬 스레드를 안전하게 종료합니다.")
            health_daemon.stop()

        for c in cams: 
            c.reader.running = False
            c.recorder.running = False
            
        if is_gui_mode:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

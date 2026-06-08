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
    """시스템 공통 설정(system_config.json)을 로드합니다."""
    default_config = {
        "terminal_id": "99999",
        "logging": {"dir": "./logs", "level": "INFO"},
        "event_config": {
            "intrusion": {"enabled": False, "cooldown_sec": 600},
            "illegal_parking": {"enabled": False, "cooldown_sec": 600, "trigger_sec": 5.0, "move_threshold_ratio": 0.1},
            "no_helmet": {"enabled": False, "cooldown_sec": 600, "blur_face": True, "trigger_sec": 3.0},
            "conveyor_crossing": {
                "enabled": False, "cooldown_sec": 600, "snapshot_mode": "crossing_moment", 
                "distance_ratio": 0.5, "min_crossing_angle": 20.0, "candidate_ttl_sec": 5.0
            },
            "signal_vehicle": {"enabled": False, "cooldown_sec": 600, "motion_threshold_ratio": 0.10}
        },
        "models": {
            "MAIN": "hanjin_cctv_ppu.dxnn",
            "FACE": "yolov8m-face_ppu.dxnn",
            "HELMET": "helmet_3cls_v8_ppu.dxnn"
        },
        "model_confidences": {
            "MAIN": 0.40,
            "FACE": 0.35,
            "HELMET": 0.45
        },
        "BATCH_SIZE": 9,
        "REC_FPS": 3,
        "REC_PRE_SEC": 3,
        "REC_POST_SEC": 4,
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
    url = "1https://tmlsafety.hudaters.net/receiver/api/v1/cctv/img"
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

def save_event_image_with_mark(frame, ip, event_type, bbox, tid, terminal_id="99999", cctv_id=1):
    """프레임에 BBox를 마킹하고 이미지를 로컬에 저장한 후 API 큐에 등록합니다."""
    if IMAGE_SAVER_POOL._work_queue.qsize() > 50:
        logger.warning("이미지 저장 큐가 포화 상태입니다. 저장을 스킵합니다.")
        return
        
    try:
        img = frame.copy()
        x1, y1, x2, y2 = map(int, bbox)
        
        # Bounding Box 렌더링
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        
        now = datetime.datetime.now()
        msg = f"{event_type} ID:{tid} {now.strftime('%H:%M:%S')}"
        
        # 텍스트가 화면 위로 벗어나지 않도록 처리
        text_y = max(20, y1 - 10)
        cv2.putText(img, msg, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        dpath = os.path.join(EVENT_ROOT_DIR, "events", ip, "images", str(event_type))
        if not os.path.exists(dpath):
            os.makedirs(dpath, exist_ok=True)
            
        fname = f"{now.strftime('%Y%m%d_%H%M%S')}_{ip}_{event_type}_{tid}.jpg"
        img_path = os.path.join(dpath, fname)
        
        h, w = frame.shape[:2]
        ai_detected_bboxes = [{"id": tid, "box": [x1, y1, x2, y2], "label": event_type}]
        
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
# [6] DeepX NPU 모델 추론 (Shape 오류 수정 및 C++ 안정성 강화)
# ==========================================
class YoLoDeepX:
    def __init__(self, engine_path):
        if not HAS_DX_ENGINE:
            raise RuntimeError("dx_engine이 설치되지 않은 서버/PC 환경에서는 YoLoDeepX(NPU) 객체를 생성할 수 없습니다.")
            
        self.engine_path = engine_path
        self.engine = None
        
        try:
            io = InferenceOption()
            self.engine = InferenceEngine(self.engine_path, io)
            
            # [수정 1] NHWC vs NCHW 레이아웃 동적 감지를 통한 정확한 해상도 추출
            input_info = self.engine.get_input_tensors_info()
            shape = input_info[0]["shape"]
            
            if len(shape) == 4:
                # 마지막 차원이 채널(1, 3, 4)인 경우 NHWC 포맷: [Batch, Height, Width, Channels]
                if shape[-1] in [1, 3, 4]:
                    self.input_height, self.input_width = shape[1], shape[2]
                # NCHW 포맷인 경우: [Batch, Channels, Height, Width]
                else:
                    self.input_height, self.input_width = shape[2], shape[3]
            elif len(shape) == 3:
                self.input_height, self.input_width = shape[0], shape[1]
            else:
                self.input_height, self.input_width = 640, 640
                
            out_info = self.engine.get_output_tensors_info()
            total_bytes = np.prod(out_info[0]["shape"])
            logger.info(f"💡 [DeepX] {os.path.basename(self.engine_path)} 로드 완료 (입력: {self.input_width}x{self.input_height}, PPU: {total_bytes} Bytes)")
                    
        except Exception as e:
            logger.error(f"[DeepX Load Fail] 엔진 초기화 실패 ({engine_path}): {e}")
            raise e

    def __del__(self):
        if hasattr(self, 'engine') and self.engine is not None:
            try:
                del self.engine
            except:
                pass
    def release(self):
        """C++ 런타임 엔진 명시적 메모리 해제"""
        if hasattr(self, 'engine') and self.engine is not None:
            try:
                del self.engine
                self.engine = None
            except:
                pass
                
    def letter_box(self, img):
        h, w = img.shape[:2]
        scale = min(self.input_height / h, self.input_width / w)
        nw, nh = int(w * scale), int(h * scale)
        
        resized = cv2.resize(img, (nw, nh))
        canvas = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)
        
        dw, dh = (self.input_width - nw) // 2, (self.input_height - nh) // 2
        canvas[dh:dh+nh, dw:dw+nw] = resized
        
        return canvas, scale, (dw, dh)

    def infer(self, img, conf_override=None):
        if img is None: 
            return np.empty((0,6))
            
        h_orig, w_orig = img.shape[:2]
        npu_input, scale, offset = self.letter_box(img)
        dw, dh = offset
        
        try:
            # [수정 2] C++ 바인딩 오류(FATAL) 방지를 위한 메모리 연속성 및 데이터 타입 보장
            input_tensor = cv2.cvtColor(npu_input, cv2.COLOR_BGR2RGB)
            input_tensor = np.expand_dims(input_tensor, axis=0)
            input_tensor = np.ascontiguousarray(input_tensor, dtype=np.uint8)
            
            output_tensor = self.engine.run([input_tensor])
            thres = conf_override if conf_override is not None else 0.40
            
            raw_data = output_tensor[0]
            if isinstance(raw_data, bytes):
                flat = np.frombuffer(raw_data, dtype=np.uint8)
            else:
                flat = np.array(raw_data, copy=False).view(np.uint8).flatten()
            
            if len(flat) == 0:
                return np.empty((0,6))
                
            stride = 0
            if len(flat) % 32 == 0:
                stride = 32
            elif len(flat) % 28 == 0:
                stride = 28
                
            if stride > 0:
                num_boxes = len(flat) // stride
                flat_stride = flat.reshape(num_boxes, stride).copy()
                
                cxcywh = np.ascontiguousarray(flat_stride[:, :16]).view(np.float32).reshape(num_boxes, 4)
                scores = np.ascontiguousarray(flat_stride[:, 20:24]).view(np.float32).flatten()
                labels = np.ascontiguousarray(flat_stride[:, 24:28]).view(np.uint32).flatten()
                
                mask = scores >= thres
                if not np.any(mask):
                    return np.empty((0,6))
                    
                cxcywh = cxcywh[mask]
                scores = scores[mask]
                labels = labels[mask]
                
                cx, cy = cxcywh[:, 0], cxcywh[:, 1]
                w, h = cxcywh[:, 2], cxcywh[:, 3]
                
                x1 = cx - w * 0.5
                y1 = cy - h * 0.5
                x2 = cx + w * 0.5
                y2 = cy + h * 0.5
                
                boxes_x1y1wh = np.column_stack([x1, y1, w, h])
                indices = cv2.dnn.NMSBoxes(boxes_x1y1wh.tolist(), scores.tolist(), thres, 0.45)
                
                if len(indices) == 0:
                    return np.empty((0,6))
                    
                keep = np.array(indices).flatten()
                x1, y1, x2, y2 = x1[keep], y1[keep], x2[keep], y2[keep]
                scores, labels = scores[keep], labels[keep]
                
                x1 = np.clip((x1 - dw) / scale, 0, w_orig)
                y1 = np.clip((y1 - dh) / scale, 0, h_orig)
                x2 = np.clip((x2 - dw) / scale, 0, w_orig)
                y2 = np.clip((y2 - dh) / scale, 0, h_orig)
                
                return np.column_stack([x1, y1, x2, y2, scores, labels])
                
            else:
                logger.error(f"⚠️ 지원하지 않는 NPU 버퍼 구조입니다: {len(flat)} bytes")
                return np.empty((0,6))
                
        except Exception as e:
            logger.error(f"NPU Inference Error: {e}")
            return np.empty((0,6))

# ==========================================
# [7] 객체 트래커 및 영상 녹화기
# ==========================================
# [1] 시스템 기본 설정 및 상수 영역 하단에 추가
DEBUG_MODE = False
class SimpleTracker:
    def __init__(self, max_lost=30): 
        self.next_id = 1
        self.tracks = {}
        self.max_lost = max_lost

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
                # 💡 [수정] bbox뿐만 아니라 conf(Confidence)도 업데이트
                self.tracks[tid].update({'bbox': detections[best_idx][:4], 'lost': 0, 'conf': detections[best_idx][4]})
                used_dets.add(best_idx)
            else: 
                self.tracks[tid]['lost'] += 1
                
        self.tracks = {tid: t for tid, t in self.tracks.items() if t['lost'] <= self.max_lost}
        
        res_tracks = []
        for i, det in enumerate(detections):
            if i not in used_dets:
                # 💡 [수정] 신규 객체 등록 시 conf 추가
                self.tracks[self.next_id] = {'bbox': det[:4], 'lost': 0, 'cls': int(det[5]), 'conf': det[4]}
                self.next_id += 1
                
        for tid, trk in self.tracks.items():
            if trk['lost'] == 0:
                # 💡 [수정] 1.0 하드코딩 대신 실제 conf 반환 (det 포맷 유지: x1, y1, x2, y2, tid, conf, cls)
                res_tracks.append([*trk['bbox'], tid, trk.get('conf', 1.0), trk['cls']])
                
        return np.array(res_tracks)

class VideoRecorder:
    def __init__(self, ip):
        self.ip = ip
        self.fps = SYS_CFG.get("REC_FPS", 3)
        self.buffer = deque(maxlen=self.fps * SYS_CFG.get("REC_PRE_SEC", 3))
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

    def trigger(self, event_name):
        now = time.time()
        post_sec = SYS_CFG.get("REC_POST_SEC", 4)
        
        if self.recording:
            self.record_end_time = now + post_sec
        else:
            logger.info(f"🎥 [녹화시작] {self.ip} - {event_name}")
            self.recording = True
            self.record_end_time = now + post_sec
            self.current_event = event_name
            
            # 프리버퍼의 모든 프레임을 기록 큐에 일괄 삽입
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
                    
                fname = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.ip}_{self.current_event}.mp4"
                fpath = os.path.join(dpath, fname)
                
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
        self.states = defaultdict(lambda: {'start_fid': 0, 'pos': None})
        
        self.trigger_sec = config.get("trigger_sec", 5.0)
        self.move_threshold_ratio = config.get("move_threshold_ratio", 0.1)
        
    def process(self, tracks, track_map, motion_mask, frame, fid, **kwargs):
        triggered = []
        curr_ids = set()
        
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
                    
                    # 신규 진입 또는 동적 임계값 이상 이동 시 초기화
                    if self.states[tid]['start_fid'] == 0 or get_distance(c, self.states[tid]['pos']) > dynamic_move_threshold:
                        self.states[tid].update({
                            'start_fid': fid, 
                            'pos': c, 
                            'bbox': t[:4], 
                            'frame': frame.copy() if frame is not None else None,
                            'fid': fid,
                            'triggered': False # 중복 알람 방지용 플래그
                        })
                    else:
                        # [핵심 보완] 정지 상태 유지 중에는 최신 스냅샷과 BBox로 지속 갱신 (API 이미지 이격 방지)
                        self.states[tid].update({
                            'bbox': t[:4],
                            'frame': frame.copy() if frame is not None else None,
                            'fid': fid
                        })
                        
                        # [가변 FPS 대응] 실제 정지 체류 시간(Duration) 계산
                        duration_sec = (fid - self.states[tid]['start_fid']) / self.fps
                        
                        if not self.states[tid].get('triggered', False) and duration_sec >= self.trigger_sec:
                            triggered.append({
                                'tid': tid, 
                                'bbox': self.states[tid]['bbox'], 
                                'frame': self.states[tid]['frame'],
                                'fid': self.states[tid]['fid']
                            })
                            self.states[tid]['triggered'] = True
                        
        # 프레임에서 사라진 객체 상태 정리
        for tid in list(self.states.keys()):
            if tid not in curr_ids: 
                del self.states[tid]
                
        return triggered

import math
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
        
        persons = [t for t in tracks if track_map.get(int(t[4])) == ID_G_PERSON]
        low_bodies = [t for t in tracks if track_map.get(int(t[4])) == ID_PERSON_LOW]
        
        for p in persons:
            p_tid = int(p[4])
            curr_ids.add(p_tid)
            
            px1, py1, px2, py2 = p[:4]
            person_height = max(1, py2 - py1)
            p_foot = (int((px1 + px2) / 2), int(py2))
            
            best_low_box = None
            max_ioa = 0
            
            for lb in low_bodies:
                lx1, ly1, lx2, ly2 = lb[:4]
                lcx, lcy = (lx1 + lx2) / 2, (ly1 + ly2) / 2
                
                if lcy < py1 + person_height * 0.4: 
                    continue
                    
                ioa = self._get_intersection_over_lowbody_area(lb[:4], p[:4])
                if ioa > max_ioa:
                    max_ioa = ioa
                    best_low_box = lb[:4]
                    
            if max_ioa >= 0.4 and best_low_box is not None:
                lx1, ly1, lx2, ly2 = best_low_box
                low_height = max(1, ly2 - ly1)
                
                curr_pos = (int((lx1 + lx2) / 2), int(ly2 - low_height * 0.1))
                
                self.lb_offsets[p_tid] = (curr_pos[0] - p_foot[0], curr_pos[1] - p_foot[1])
                self.lb_last_height[p_tid] = low_height
                event_bbox = tuple(best_low_box)
            else:
                if p_tid in self.lb_offsets:
                    ox, oy = self.lb_offsets[p_tid]
                    curr_pos = (p_foot[0] + ox, p_foot[1] + oy)
                    low_height = self.lb_last_height.get(p_tid, person_height * 0.4)
                    event_bbox = (px1, py2 - low_height, px2, py2)
                else: 
                    continue

            if p_tid in self.prev:
                jump_dist = get_distance(self.prev[p_tid], curr_pos)
                if jump_dist > person_height * 0.4:
                    del self.prev[p_tid]
                    self.prev[p_tid] = curr_pos
                    continue

            # 후보 등록 시점 (선분 교차 순간의 증거 스냅샷 보존)
            if p_tid in self.prev and p_tid not in self.candidates:
                trajectory = (self.prev[p_tid], curr_pos)
                
                for p1, p2 in self.lines:
                    if self._is_intersect(p1, p2, trajectory[0], trajectory[1]):
                        cross_angle = self._get_angle_between_lines((p1, p2), trajectory)
                        
                        if cross_angle >= self.min_crossing_angle:
                            self.candidates[p_tid] = {
                                'person_height': person_height, 
                                'timestamp_fid': fid, 
                                'line': (p1, p2), 
                                'entry_side': ccw(p1, p2, trajectory[0]), 
                                'bbox': event_bbox, 
                                'frame': frame.copy() if frame is not None else None,
                                'fid': fid
                            }
                        break
            
            # 최종 판정 로직
            if p_tid in self.candidates:
                cand = self.candidates[p_tid]
                p1, p2 = cand['line']
                curr_side = ccw(p1, p2, curr_pos)
                
                if cand['entry_side'] != 0 and curr_side != 0 and cand['entry_side'] != curr_side:
                    perp_dist = self._get_perpendicular_distance(p1, p2, curr_pos)
                    
                    dx = abs(p2[0] - p1[0])
                    dy = abs(p2[1] - p1[1])
                    line_tilt_angle = math.degrees(math.atan2(dy, dx))
                    
                    tilt_factor = 1.0 + (math.sin(math.radians(line_tilt_angle)) * 0.5)
                    dynamic_threshold = cand['person_height'] * self.distance_ratio * tilt_factor
                    
                    if perp_dist >= dynamic_threshold:
                        # [핵심 보완] 완전히 넘어갔음을 확정한 후, API에는 '교차 순간(cand)'의 이미지와 BBox를 발송
                        triggered.append({
                            'tid': p_tid, 
                            'bbox': cand['bbox'], 
                            'frame': cand['frame'],
                            'fid': cand['fid']
                        })
                        del self.candidates[p_tid]
                        
                elif (fid - cand['timestamp_fid']) / self.fps > self.candidate_ttl_sec: 
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
        
        self.trigger_sec = config.get("trigger_sec", 3.0)
        self.window_sec = config.get("window_sec", 30.0)
        
        # FPS에 따라 window 길이를 fid 단위로 변환 (타임아웃용)
        self.window_fids = int(self.window_sec * self.fps)
        
        # 가변 FPS 환경에서 단순 1~2프레임 튀는 오탐을 막기 위한 절대적인 '최소 감지 요구 프레임 수'
        self.min_hit_count = config.get("min_hit_count", 3)

    def _get_intersection_over_head_area(self, head_box, person_box):
        """머리 면적 대비 교차 면적 비율 계산"""
        inter_w = max(0, min(head_box[2], person_box[2]) - max(head_box[0], person_box[0]))
        inter_h = max(0, min(head_box[3], person_box[3]) - max(head_box[1], person_box[1]))
        inter_area = inter_w * inter_h
        
        head_area = max(1, (head_box[2] - head_box[0]) * (head_box[3] - head_box[1]))
        return inter_area / head_area

    def process(self, tracks, track_map, motion_mask, frame, fid, **kwargs):
        triggered = []
        helmet_tracks = kwargs.get('helmet_tracks', [])
        
        unhelmeted_heads = [t for t in helmet_tracks if int(t[6]) == ID_H_NO_HELMET]
        current_nh_persons = []
        
        for p in tracks:
            p_tid = int(p[4])
            if track_map.get(p_tid) != ID_G_PERSON: 
                continue
                
            # ROI 설정 시, 사람의 발 위치가 ROI 내부에 있는지 검사
            if self.roi_poly.size > 0:
                foot_pt = get_foot_point(*p[:4])
                if cv2.pointPolygonTest(self.roi_poly, foot_pt, False) < 0:
                    continue
                    
            px1, py1, px2, py2 = p[:4]
            person_height = max(1, py2 - py1)
            person_width = max(1, px2 - px1)

            max_ioa = 0
            nh_box_match = None
            
            for head in unhelmeted_heads:
                hx1, hy1, hx2, hy2 = head[:4]
                hcx, hcy = (hx1 + hx2) / 2, (hy1 + hy2) / 2

                # [해부학적 필터 1] 머리는 전신의 상단 40% 이내
                if hcy > py1 + person_height * 0.4:
                    continue
                    
                # [수평 필터 2] 머리는 전신의 좌우 폭(15% 마진) 이내
                margin = person_width * 0.15
                if hcx < px1 - margin or hcx > px2 + margin:
                    continue

                ioa = self._get_intersection_over_head_area(head[:4], p[:4])
                if ioa > max_ioa: 
                    max_ioa = ioa
                    nh_box_match = head[:4]
                    
            # 헬멧 미착용으로 판별된 객체 수집
            if max_ioa >= 0.5 and nh_box_match is not None:
                current_nh_persons.append({
                    'tid': p_tid,
                    'head_bbox': nh_box_match,
                    'person_bbox': p[:4]
                })
                
        # 수집된 객체들을 세션에 매칭 및 업데이트
        for nh_p in current_nh_persons:
            matched_session = None
            
            for session in self.sessions:
                # Track ID 유지 또는 공간적 겹침(IoU) 기반 세션 매칭
                if session['last_tid'] == nh_p['tid'] or calculate_iou(nh_p['person_bbox'], session['last_person_bbox']) > 0.3:
                    matched_session = session
                    break
                    
            if matched_session:
                matched_session['hit_fids'].add(fid)
                matched_session['last_tid'] = nh_p['tid']
                matched_session['last_person_bbox'] = nh_p['person_bbox']
                matched_session['bbox'] = nh_p['head_bbox']
                matched_session['frame'] = frame.copy() if frame is not None else None
                matched_session['fid'] = fid
            else:
                self.sessions.append({
                    'start_fid': fid,
                    'hit_fids': {fid},
                    'last_tid': nh_p['tid'],
                    'last_person_bbox': nh_p['person_bbox'],
                    'bbox': nh_p['head_bbox'],
                    'frame': frame.copy() if frame is not None else None,
                    'fid': fid,
                    'triggered': False
                })

        # 세션 검사 (트리거 판별 및 정리)
        active_sessions = []
        for session in self.sessions:
            # 30초 윈도우가 경과한 세션은 폐기
            if fid - session['start_fid'] > self.window_fids:
                continue
                
            # 현재 프레임과 첫 감지 프레임 간의 '비디오 기준 체류 시간' 계산
            duration_sec = (fid - session['start_fid']) / self.fps
            
            # 가변 FPS 환경 트리거 조건: 3초 이상 경과 & 1프레임 노이즈가 아님(3회 이상 감지)
            if not session['triggered'] and duration_sec >= self.trigger_sec and len(session['hit_fids']) >= self.min_hit_count:
                triggered.append({
                    'tid': session['last_tid'], 
                    'bbox': session['bbox'], 
                    'frame': session['frame'], 
                    'fid': session['fid']
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
        self.motion_ratio = config.get("motion_threshold_ratio", 0.10)
        
    def process(self, tracks, track_map, motion_mask, frame, fid, **kwargs):
        triggered = []
        curr_ids = set()
        
        if self.roi_poly.size == 0 or motion_mask is None: 
            return triggered
            
        ppts = [get_foot_point(*t[:4]) for t in tracks if track_map.get(int(t[4])) == ID_G_PERSON]
        scale_x = 640 / SCREEN_WIDTH
        scale_y = 360 / SCREEN_HEIGHT
        
        for t in tracks:
            tid = int(t[4])
            if track_map.get(tid) not in TARGET_VEHICLES: 
                continue
                
            curr_ids.add(tid)
            x1, y1, x2, y2 = t[:4]
            fc = get_foot_point(*t[:4])
            v_size = max(x2 - x1, y2 - y1)
            
            if len(self.history[tid]) > 0 and get_distance(self.history[tid][-1], fc) > v_size * 0.6: 
                self.history[tid].clear()
                continue
                
            self.history[tid].append(fc)
            h_list = list(self.history[tid])
            
            if len(h_list) > 5:
                start_p = (sum(p[0] for p in h_list[:3])/3, sum(p[1] for p in h_list[:3])/3)
                end_p = (sum(p[0] for p in h_list[-3:])/3, sum(p[1] for p in h_list[-3:])/3)
                dist = get_distance(start_p, end_p)
                
                if dist >= v_size * 0.15 and cv2.pointPolygonTest(self.roi_poly, get_center_point(*t[:4]), False) >= 0:
                    mx1 = max(0, int(x1 * scale_x))
                    my1 = max(0, int(y1 * scale_y))
                    mx2 = min(640, int(x2 * scale_x))
                    my2 = min(360, int(y2 * scale_y))
                    
                    if mx2 > mx1 and my2 > my1:
                        car_roi = motion_mask[my1:my2, mx1:mx2]
                        _, m_only = cv2.threshold(car_roi, 250, 255, cv2.THRESH_BINARY)
                        
                        total_px = (mx2 - mx1) * (my2 - my1)
                        if total_px > 0 and (cv2.countNonZero(m_only) / total_px) > self.motion_ratio:
                            has_signalman = any(
                                math.sqrt(max(t[0]-pt[0], 0, pt[0]-t[2])**2 + max(t[1]-pt[1], 0, pt[1]-t[3])**2) < v_size * 1.5 
                                for pt in ppts
                            )
                            
                            if not has_signalman:
                                triggered.append({
                                    'tid': tid, 
                                    'bbox': t[:4], 
                                    'frame': frame.copy() if frame is not None else None,
                                    'fid': fid
                                })
                                self.history[tid].clear()
                                
        for tid in list(self.history.keys()):
            if tid not in curr_ids: 
                del self.history[tid]
                
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
    def __init__(self, ip, conf, det_main, det_helmet, det_face, cam_id):
        self.ip = ip
        self.conf = conf
        self.cam_id = cam_id
        self.events = conf.get('events', [])
        
        self.det_main = det_main
        self.det_helmet = det_helmet
        self.det_face = det_face
        
        self.trk_main = SimpleTracker()
        self.trk_helmet = SimpleTracker()
        
        self.reader = FrameReader(conf.get('url', ''), ip)
        self.recorder = VideoRecorder(ip)
        self.motion_det = MotionDetector()
        
        self.alerted = defaultdict(set)
        self.last_evt_t = {}
        self.visual_alarms = {}

        from collections import deque
        self.fps_queue = deque(maxlen=30)
        self.current_fps = 0.0

        self.roi_poly_norm = conf.get('roi_poly_norm', [])
        self.roi_lines_norm = conf.get('roi_lines_norm', [])
        self.roi_poly = []
        self.roi_lines = []
        self.roi_frame_shape = None # 해상도 변경 감지용
        
        self.handlers = {}
        for ename in self.events:
            if ename in EVENT_REGISTRY:
                self.handlers[ename] = EVENT_REGISTRY[ename](SYS_CFG.get("event_config", {}).get(ename, {}), self.roi_poly, self.roi_lines)

    def update_config(self, new_conf):
        """웹 UI 등 외부에서 cameras.json이 변경되었을 때, 무중단으로 설정을 핫 리로드합니다."""
        old_events = self.events.copy()
        self.events = new_conf.get('events', [])
        self.roi_poly_norm = new_conf.get('roi_poly_norm', [])
        self.roi_lines_norm = new_conf.get('roi_lines_norm', [])
        
        # 해상도 변경 감지 변수를 초기화하여 다음 프레임에서 ROI 역정규화를 강제 수행
        self.roi_frame_shape = None 
        
        # 이벤트 핸들러 재구성
        self.handlers = {}
        for ename in self.events:
            if ename in EVENT_REGISTRY:
                # 새로운 설정과 텅 빈 ROI로 초기화 (이후 process_frame 타임에 runtime_roi가 자동 주입됨)
                event_cfg = SYS_CFG.get("event_config", {}).get(ename, {})
                self.handlers[ename] = EVENT_REGISTRY[ename](event_cfg, [], [])
                
        logger.info(f"🔄 [CAM:{self.ip}] 무중단 설정 리로드 완료: {old_events} -> {self.events}")

    def _update_runtime_roi(self, frame_shape):
        if self.roi_frame_shape == frame_shape[:2]:
            return
            
        height, width = frame_shape[:2]
        if self.roi_poly_norm:
            self.roi_poly = denormalize_roi_points(self.roi_poly_norm, width, height)
        if self.roi_lines_norm:
            self.roi_lines = denormalize_roi_points(self.roi_lines_norm, width, height)

        # ROI가 스케일링 되었으므로, 이벤트 핸들러에도 새 좌표를 주입하여 갱신합니다.
        for ename in self.events:
            if ename in EVENT_REGISTRY:
                event_cfg = SYS_CFG.get("event_config", {}).get(ename, {})
                self.handlers[ename] = EVENT_REGISTRY[ename](event_cfg, self.roi_poly, self.roi_lines)
                
        self.roi_frame_shape = frame_shape[:2]

    def process_frame(self):
        fr, fid, connected = self.reader.read()
        if fr is not None: 
            self.recorder.update(fr)
        return fr, fid, connected

    def apply_face_blur(self, frame, bbox):
        if frame is None or self.det_face is None: 
            return frame
            
        blur_img = frame.copy()
        try:
            face_conf = SYS_CFG.get("model_confidences", {}).get("FACE", 0.35)
            f_dets = self.det_face.infer(blur_img, conf_override=face_conf)
            
            for f in f_dets:
                fx1, fy1, fx2, fy2 = map(int, f[:4])
                fh, fw = fy2 - fy1, fx2 - fx1
                
                # 터무니없는 크기의 오탐 얼굴 방어 (화면의 80% 이상)
                if fw > blur_img.shape[1] * 0.8: 
                    continue 
                    
                roi = blur_img[fy1:fy2, fx1:fx2]
                if roi.size > 0:
                    small = cv2.resize(roi, (max(1, fw//15), max(1, fh//15)), interpolation=cv2.INTER_LINEAR)
                    blur_img[fy1:fy2, fx1:fx2] = cv2.resize(small, (fw, fh), interpolation=cv2.INTER_NEAREST)
        except Exception as e: 
            logger.error(f"모자이크 처리 실패: {e}")
            
        return blur_img

    def run_logic(self, fr, fid, d_main_res, d_helmet_res):

        if fr is None:
            return [], [], {}

        # [추가] 이동 평균 FPS 계산 로직
        now_t = time.time()
        self.fps_queue.append(now_t)
        if len(self.fps_queue) > 1:
            time_diff = self.fps_queue[-1] - self.fps_queue[0]
            self.current_fps = len(self.fps_queue) / time_diff if time_diff > 0 else 0.0

        self._update_runtime_roi(fr.shape)
        motion_mask = self.motion_det.apply(fr)

        # 메인 트래커 업데이트
        d_main_filtered = [d for d in d_main_res if int(d[5]) not in [ID_H_HELMET, ID_H_NO_HELMET]]
        t_main = self.trk_main.update(d_main_filtered)
        
        # 헬멧 트래커 업데이트
        d_helmet_filtered = [d for d in d_helmet_res if int(d[5]) == ID_H_NO_HELMET]
        t_helmet = self.trk_helmet.update(d_helmet_filtered)

        now = time.time()
        current_alarms = {} 
        track_map_main = {int(t[4]): int(t[6]) for t in t_main}

        for ename, handler in self.handlers.items():
            kwargs = {'helmet_tracks': t_helmet} if ename == "no_helmet" else {}
            
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
                
                if ename not in self.alerted[tid] and (now - self.last_evt_t.get(ename, 0) >= cooldown):
                    
                    # 💡 [수정] 이벤트 상세 컨텍스트 수집 및 로깅
                    conf_val = next((float(t[5]) for t in t_main if int(t[4]) == tid), 0.0)
                    cls_id = track_map_main.get(tid, -1)
                    terminal_id = SYS_CFG.get("terminal_id", "99999")
                    
                    # ROI 정보 압축 (좌표계가 너무 길어지는 것 방지)
                    roi_str = f"Poly[{len(self.roi_poly)} pts]" if self.roi_poly else "None"
                    
                    log_msg = (
                        f"🔥 [EVENT TRIGGERED] CAM:{self.cam_id}({self.ip}) | Event:{ename} | "
                        f"TermID:{terminal_id} | TID:{tid} | ClassID:{cls_id} | Conf:{conf_val:.2f} | "
                        f"Trace(BBox):{tuple(map(int, bbox))} | FPS:{self.current_fps:.1f} | ROI:{roi_str}"
                    )
                    logger.warning(log_msg)
                    
                    blur_face_option = SYS_CFG.get("event_config", {}).get(ename, {}).get("blur_face", True)
                    saved_img = self.apply_face_blur(ev_frame, bbox) if blur_face_option else ev_frame
                    
                    save_event_image_with_mark(
                        frame=saved_img, ip=self.ip, event_type=ename, bbox=bbox, tid=tid, 
                        terminal_id=terminal_id, cctv_id=self.cam_id
                    )
                    
                    self.recorder.trigger(ename)
                    self.alerted[tid].add(ename)
                    self.last_evt_t[ename] = now
                    
                current_alarms[tid] = ename
        
        # 화면 시각적 알람 처리
        alarm_duration = SYS_CFG.get("VISUAL_ALARM_DURATION", 5.0)
        for tid, ename in current_alarms.items(): 
            self.visual_alarms[tid] = {'evt': ename, 'expire': now + alarm_duration}
            
        for tid in list(self.visual_alarms.keys()):
            if now > self.visual_alarms[tid]['expire']: 
                del self.visual_alarms[tid]
                
        return t_main, t_helmet, {t: info['evt'] for t, info in self.visual_alarms.items()}

    def draw(self, fr, t_main, t_helmet, alarms, connected=True):
        if fr is None or not connected:
            blank = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(blank, f"CAM {self.cam_id} NO SIGNAL", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.putText(blank, self.ip, (50, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            return blank

        h_frame, w_frame = fr.shape[:2]
        
        # 화면 테두리 알람 마킹
        if len(alarms) > 0: 
            cv2.rectangle(fr, (0, 0), (w_frame, h_frame), (0, 0, 255), 20)
            
        # ROI 다각형 및 라인 렌더링
        if len(self.roi_poly) > 2: 
            cv2.polylines(fr, [np.array(self.roi_poly, np.int32)], True, (0, 255, 255), 2)
        if self.roi_lines:
            for i in range(0, len(self.roi_lines), 2):
                if i + 1 < len(self.roi_lines): 
                    cv2.line(fr, tuple(self.roi_lines[i]), tuple(self.roi_lines[i+1]), (0, 0, 255), 2)

        # Main Tracker BBox 렌더링
        for t in t_main:
            tid = int(t[4])
            cls_id = int(t[6])
            color = (0, 255, 0)
            
            if cls_id == ID_G_PERSON: label = f"Person [{tid}]"
            elif cls_id == ID_PERSON_LOW: label, color = f"LowBody [{tid}]", (0, 150, 0)
            elif cls_id == ID_REFLECTIVE_VEST: label, color = f"Signalman [{tid}]", (0, 255, 255)
            elif cls_id in TARGET_VEHICLES: label, color = f"Vehicle [{tid}]", (255, 100, 0)
            else: label = f"OBJ [{tid}]"

            if tid in alarms: 
                color = (0, 0, 255)
                label = f"ALARM: {label}"
                
            thickness = 3 if tid in alarms else 2
            cv2.rectangle(fr, (int(t[0]), int(t[1])), (int(t[2]), int(t[3])), color, thickness)
            cv2.putText(fr, label, (int(t[0]), int(t[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Helmet Tracker BBox 렌더링
        for t in t_helmet:
            tid = int(t[4])
            color = (0, 0, 255)
            label = f"Head [{tid}]"
            
            thickness = 3 if tid in alarms else 2
            cv2.rectangle(fr, (int(t[0]), int(t[1])), (int(t[2]), int(t[3])), color, thickness)
            cv2.putText(fr, label, (int(t[0]), int(t[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.rectangle(fr, (0, 0), (220, 100), (0, 0, 0), -1) 
        cv2.putText(fr, f"CAM {self.cam_id}", (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        
        # FPS 수치에 따라 색상 변경 (10 미만이면 붉은색 경고)
        fps_color = (0, 255, 0) if self.current_fps >= 10.0 else (0, 0, 255)
        cv2.putText(fr, f"FPS: {self.current_fps:.1f}", (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.0, fps_color, 2)
        
        active_alarms = set(alarms.values())
        menu_height = len(self.events) * 40 + 10
        
        overlay = fr.copy()
        cv2.rectangle(overlay, (w_frame - 250, 0), (w_frame, menu_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, fr, 0.5, 0, fr)
        
        y_pos = 35
        for evt in self.events:
            display_name = EVENT_REGISTRY[evt].gui_name if evt in EVENT_REGISTRY else evt.upper()
            color = (0, 0, 255) if evt in active_alarms else (0, 255, 0)
            prefix = "[!] " if evt in active_alarms else " -  "
            
            cv2.putText(fr, f"{prefix}{display_name}", (w_frame - 240, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            y_pos += 40
            
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
    def __init__(self, terminal_id, version="v1.1.0", interval_sec=300):
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
# [핵심 1] 추론 및 모션 감지 로직을 병렬 스레드로 실행하기 위한 래퍼 함수
def process_camera_task(idx, cam, fr, fid, connected, main_conf, helmet_conf):
    if not connected or fr is None:
        return idx, None, None, None, None, False
                
    # 전달받은 cam 객체를 통해 NPU 추론 실행
    d_main_res = cam.det_main.infer(fr, conf_override=main_conf)
    
    d_helmet_res = []
    if "no_helmet" in cam.events:
        d_helmet_res = cam.det_helmet.infer(fr, conf_override=helmet_conf)
        
    return idx, fr, fid, d_main_res, d_helmet_res, True        
# ==========================================
# [12] 메인 프로세스 
# ==========================================
def main():
    global DEBUG_MODE
    logger.info("[System] 단일 스크립트 기반 YOLOv8 모듈화 시스템 초기화 완료")
    
    rtsp_list = load_rtsp_list_from_csv(CAMERA_LIST_FILE)
    if not rtsp_list:
        logger.error(f"카메라 목록 파일({CAMERA_LIST_FILE})을 확인하십시오.")
        return
        
    config_file = os.path.join(PROJECT_ROOT, "cameras.json")
    camera_configs = {}
    
    # 💡 [수정] 디버그 모드 및 초기화 질문 (디폴트 'n')
    debug_ans = input(">> 디버그 모드를 활성화하시겠습니까? (상세 로그 출력) [y/N]: ").strip().lower()
    DEBUG_MODE = True if debug_ans == 'y' else False
    if DEBUG_MODE:
        logger.setLevel(logging.DEBUG)
        logger.debug("🛠️ 디버그 모드가 활성화되었습니다. 상세 로깅이 시작됩니다.")
    
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r', encoding='utf-8') as f: 
                camera_configs = json.load(f)
        except Exception as e: 
            logger.error(f"cameras.json 로드 실패: {e}")
            pass
            
        # 💡 [수정] 아무것도 입력 안하면 기본적으로 'n'으로 처리되도록 방어 로직
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
        logger.info("DeepX 모델을 VPU 메모리로 할당 중...")
        
        # [클린업] 외부 SDK (factory, common) 의존성을 모두 삭제했습니다.
        d_main = YoLoDeepX(SYS_CFG["models"]["MAIN"])
        d_face = YoLoDeepX(SYS_CFG["models"]["FACE"])
        d_helmet = YoLoDeepX(SYS_CFG["models"]["HELMET"])
        
        logger.info(f"✅ DeepX 모델 로드 및 SDK 독립형 고속 파서 바인딩 성공")

    except Exception as e:
        logger.error(f"모델 로드 중 오류 발생: {e}")
        return

    cams = []
    for i, rtsp in enumerate(rtsp_list):
        ip = extract_ip(rtsp)
        conf = camera_configs.get(ip)
        
        if not conf or not conf.get('events'): 
            continue
            
        conf['url'] = rtsp
        cams.append(Camera(ip, conf, d_main, d_helmet, d_face, cam_id=i+1))
        logger.info(f"Loaded [CAM {i+1}]: {ip}")

    # 환경 변수 스로틀링 기준
    target_fps = SYS_CFG.get("REC_FPS", 15)
    main_conf = SYS_CFG["model_confidences"]["MAIN"]
    helmet_conf = SYS_CFG["model_confidences"]["HELMET"]
    loop_count = 0
    
    # [추가] 5분(300초) 간격의 헬스 체크 데몬 실행
    terminal_id = SYS_CFG.get("terminal_id", "99999")
    software_version = "v1.1.0"  # 필요 시 SYS_CFG에서 로드 가능
    health_daemon = HealthCheckDaemon(terminal_id=terminal_id, version=software_version, interval_sec=300)
    
    # [추가] 핫 리로드를 위한 설정 파일 타임스탬프 기록
    last_config_mtime = 0
    if os.path.exists(config_file):
        last_config_mtime = os.path.getmtime(config_file)
        
    # [추가] 웹 UI 스냅샷 공유를 위한 RAM 디스크 폴더 준비 (리눅스 기준)
    RAM_DISK_DIR = "/dev/shm/cctv_frames"
    if not os.path.exists(RAM_DISK_DIR):
        try: os.makedirs(RAM_DISK_DIR, exist_ok=True)
        except: RAM_DISK_DIR = "./web_frames" # 윈도우나 지원하지 않는 환경용 폴백

    try:
        while True:
            start_time = time.time()
            
            # [추가] 3초(대략 45루프)마다 설정 파일의 변경 여부를 검사 (Watchdog)
            if loop_count % 45 == 0 and os.path.exists(config_file):
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
                    except Exception as e:
                        logger.error(f"핫 리로드 중 예외 발생 (JSON 문법 오류 등): {e}")

            cpu_usage = psutil.cpu_percent(interval=None)
            if cpu_usage > 85: 
                target_fps = max(5, target_fps - 2)
            elif cpu_usage < 60: 
                target_fps = min(15, target_fps + 1)
                
            dynamic_delay = 1.0 / target_fps
            
            loop_count += 1
            if loop_count % 300 == 0: 
                gc.collect()
                mem_usage = psutil.virtual_memory().percent
                q_size = IMAGE_SAVER_POOL._work_queue.qsize()
                
                if mem_usage > 80 or q_size > 20:
                    logger.warning(f"⚠️ [System Health] CPU: {cpu_usage}% | Mem: {mem_usage}% | API Queue: {q_size}")
            
            raw_data = [c.process_frame() for c in cams]
            final_imgs = [None] * len(cams)

            # [핵심 2] ThreadPoolExecutor를 이용해 9대의 카메라 추론을 동시에 NPU로 오프로드
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(cams)) as executor:
                futures = []
                for idx, res in enumerate(raw_data):
                    fr, fid, connected = res
                    
                    if connected and fr is not None and loop_count % 100 == 0:
                        try:
                            small_fr = cv2.resize(fr, (640, 360))
                            save_path = os.path.join(RAM_DISK_DIR, f"{cams[idx].ip}.jpg")
                            cv2.imwrite(save_path, small_fr, [cv2.IMWRITE_JPEG_QUALITY, 70])
                        except Exception:
                            pass
                            
                    if not cams[idx].events:
                        if connected and fr is not None:
                            final_imgs[idx] = cams[idx].draw(fr, [], [], {}, True)
                        else:
                            final_imgs[idx] = cams[idx].draw(None, [], [], {}, False)
                        continue
                        
                    # [수정] 필요한 모든 객체를 인자로 명시적으로 전달 (NameError 방지)
                    futures.append(executor.submit(
                        process_camera_task, idx, cams[idx], fr, fid, connected, main_conf, helmet_conf
                    ))
                
                # 병렬 추론 결과 수집 및 논리(Logic) 실행
                for future in concurrent.futures.as_completed(futures):
                    idx, fr, fid, d_main_res, d_helmet_res, is_valid = future.result()
                    
                    if not is_valid:
                        final_imgs[idx] = cams[idx].draw(None, [], [], {}, False)
                        continue
                        
                    t_main, t_helmet, alarms = cams[idx].run_logic(fr, fid, d_main_res, d_helmet_res)
                    final_imgs[idx] = cams[idx].draw(fr, t_main, t_helmet, alarms, True)
            
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
        # [추가] 프로세스 종료 시 C++ 엔진 명시적 파괴 (Aborted 코어덤프 방어)
        logger.info("🔄 [SYSTEM] NPU 메모리를 안전하게 해제합니다...")
        if 'd_main' in locals(): d_main.release()
        if 'd_face' in locals(): d_face.release()
        if 'd_helmet' in locals(): d_helmet.release()
        
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

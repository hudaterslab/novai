#!/bin/bash

# 1. 사용자 및 경로 설정
ACTUAL_USER=${SUDO_USER:-$USER}
USER_HOME=$(getent passwd "$ACTUAL_USER" | cut -d: -f6)
PROJECT_DIR=$(pwd)

SERVICE_NAME="cctv_ai.service"
SERVICE_PATH="/etc/systemd/system/$SERVICE_NAME"
CAM_CONFIG_FILE="$PROJECT_DIR/cameras.json"
SYS_CONFIG_FILE="$PROJECT_DIR/system_config.json"
TARGET_SCRIPT="$PROJECT_DIR/multi_event.py"
DX_DIR="$USER_HOME/dx-runtime"

echo "====================================================="
echo " Raspberry Pi Edge AI CCTV 서비스 하이브리드 설치"
echo "====================================================="

if [ ! -f "$TARGET_SCRIPT" ]; then
    echo "❌ [설치 중단] 현재 폴더에 'multi_event.py' 파일이 존재하지 않습니다."
    exit 1
fi

# 2. system_config.json 부재 시 자동 생성
if [ ! -f "$SYS_CONFIG_FILE" ]; then
    echo "⚙️ [설정 생성] 'system_config.json' 파일을 생성합니다."
    read -p ">> 이 장비의 고유 Terminal ID를 입력하세요 (예: 10001): " USER_TERM_ID
    if [ -z "$USER_TERM_ID" ]; then USER_TERM_ID="99999"; fi
    
    sudo -u "$ACTUAL_USER" bash -c "cat > $SYS_CONFIG_FILE" << EOL
{
    "terminal_id": "$USER_TERM_ID",
    "logging": {"dir": "./logs", "level": "INFO"},
    "event_config": {
        "intrusion": {"enabled": false, "cooldown_sec": 600},
        "illegal_parking": {"enabled": false, "cooldown_sec": 600, "trigger_sec": 5.0, "move_threshold_ratio": 0.1},
        "no_helmet": {"enabled": false, "cooldown_sec": 600, "blur_face": true, "trigger_sec": 3.0},
        "conveyor_crossing": {"enabled": false, "cooldown_sec": 600, "snapshot_mode": "crossing_moment", "distance_ratio": 0.5, "min_crossing_angle": 20.0, "candidate_ttl_sec": 5.0},
        "signal_vehicle": {"enabled": false, "cooldown_sec": 600, "motion_threshold_ratio": 0.10}
    },
    "models": {
        "MAIN": "hanjin_cctv.dxnn",
        "FACE": "yolov8m-face.dxnn",
        "HELMET": "helmet_3cls_v8.dxnn"
    },
    "model_confidences": {"MAIN": 0.6, "FACE": 0.35, "HELMET": 0.55},
    "BATCH_SIZE": 9, "REC_FPS": 3, "REC_PRE_SEC": 10, "REC_POST_SEC": 10, "VISUAL_ALARM_DURATION": 5.0
}
EOL
    echo "✅ 기본 설정 생성이 완료되었습니다."
fi

echo "-----------------------------------------------------"
echo " 3. Python 환경 및 필수 패키지 설치"
echo "-----------------------------------------------------"
# python 명령어를 python3로 연결 (이미 존재해도 오류 안 나도록 -f 옵션 추가)
echo "-> 'python' 심볼릭 링크를 생성합니다..."
sudo ln -sf /usr/bin/python3 /usr/bin/python

# 필수 파이썬 패키지 설치 (최신 우분투/데비안의 PEP-668 제약 우회를 위해 break-system-packages 옵션 추가 대비)
echo "-> 필수 파이썬 패키지를 설치합니다 (pytz, psutil, requests, opencv-python)..."
python -m pip install pytz psutil requests opencv-python || \
python -m pip install pytz psutil requests opencv-python --break-system-packages

echo "-----------------------------------------------------"
echo " 4. DeepX 드라이버 및 런타임(.deb) 글로벌 설치"
echo "-----------------------------------------------------"
if ! python3 -c 'import dx_engine' > /dev/null 2>&1; then
    echo "-> 시스템에서 'dx_engine'을 찾을 수 없어 공식 패키지를 다운로드합니다..."
    DOWNLOAD_DIR="$USER_HOME/Downloads"
    sudo -u "$ACTUAL_USER" mkdir -p "$DOWNLOAD_DIR"

    sudo -u "$ACTUAL_USER" wget -q --show-progress -P "$DOWNLOAD_DIR" https://github.com/DEEPX-AI/dx_rt_npu_linux_driver/raw/refs/heads/main/release/2.4.0/dxrt-driver-dkms_2.4.0-2_all.deb
    sudo -u "$ACTUAL_USER" wget -q --show-progress -P "$DOWNLOAD_DIR" https://github.com/DEEPX-AI/dx_rt/raw/refs/heads/main/release/3.3.2/libdxrt_3.3.2_all.deb

    sudo apt install -y "$DOWNLOAD_DIR/dxrt-driver-dkms_2.4.0-2_all.deb"
    sudo apt install -y "$DOWNLOAD_DIR/libdxrt_3.3.2_all.deb"
    
    # 설치 직후 서비스 강제 재시작 (장치 인식 유도)
    sudo systemctl restart dxrt.service 2>/dev/null || true
else
    echo "✅ [확인] dx_engine (런타임)이 이미 글로벌 환경에 설치되어 있습니다."
fi

echo "-----------------------------------------------------"
echo " 5. 펌웨어(dx_fw) 전용 GitHub 클론 및 플래싱"
echo "-----------------------------------------------------"
if [ ! -d "$DX_DIR" ]; then
    echo "-> 펌웨어 파일을 가져오기 위해 저장소를 클론합니다..."
    sudo -u "$ACTUAL_USER" git clone --depth 1 https://github.com/DEEPX-AI/dx-runtime.git "$DX_DIR"
fi

if command -v dxrt-cli &> /dev/null; then
    echo "-> M.2 / PCIe 기반 펌웨어(FW) 업데이트를 시도합니다..."
    dxrt-cli -u "$DX_DIR/dx_fw/m1/latest/mdot2/fw.bin" || echo "⚠️ [안내] 펌웨어 업데이트 건너뜀 (이미 최신이거나 재부팅 필요)"
    dxrt-cli -u "$DX_DIR/dx_fw/m1m/latest/mdot2/fw.bin" > /dev/null 2>&1 || true
else
    echo "⚠️ 'dxrt-cli' 명령어를 찾을 수 없어 펌웨어 업데이트를 건너뜁니다."
fi

echo "-----------------------------------------------------"
echo " 6. dx_engine 최종 검증"
echo "-----------------------------------------------------"
if ! python3 -c 'import dx_engine' > /dev/null 2>&1; then
    echo "❌ [에러] 드라이버 설치 후에도 모듈을 로드할 수 없습니다."
    echo "   >> NPU 장치가 커널에 등록되지 않았습니다."
    echo "   >> 지금 즉시 'sudo reboot' 명령어로 PC를 재부팅한 뒤, 이 스크립트를 다시 실행해 주세요!"
    exit 1
else
    echo "✅ [검증 성공] 로컬(Global) 파이썬 환경에서 'dx_engine'이 완벽하게 로드됩니다."
fi

echo "-----------------------------------------------------"
echo " 7. Systemd 백그라운드 서비스 등록"
echo "-----------------------------------------------------"
sudo bash -c "cat > $SERVICE_PATH" << EOL
[Unit]
Description=Raspberry Pi Edge AI CCTV Event Detection
Requires=dxrt.service
After=network.target dxrt.service graphical.target
ConditionPathExists=$SYS_CONFIG_FILE
ConditionPathExists=$CAM_CSV_FILE

[Service]
Type=simple
User=$ACTUAL_USER
WorkingDirectory=$PROJECT_DIR
ExecStartPre=/bin/sleep 20
ExecStart=/bin/bash -c 'yes n | python3 -u multi_event.py'
Restart=always
RestartSec=15
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=graphical.target
EOL

sudo systemctl daemon-reload
sudo systemctl enable $SERVICE_NAME
sudo systemctl stop $SERVICE_NAME

CAM_CSV_FILE="$PROJECT_DIR/cameras.csv"
if [ ! -f "$CAM_CSV_FILE" ]; then
    echo "⚠️ 'cameras.csv' (카메라 설정 파일)이 없어 서비스가 대기 모드로 진입합니다."
else
    sudo systemctl start $SERVICE_NAME
    echo "▶️ CCTV AI 백그라운드 서비스가 구동되었습니다."
fi

echo "====================================================="
echo " 🎉 설치 완료! 쾌적하고 안정적인 Global 환경으로 설정되었습니다."
echo " ※ 만약 AI 인퍼런스가 정상 작동하지 않는다면, PC를 1회 껐다 켜주세요 (Cold Boot 권장)."
echo "====================================================="
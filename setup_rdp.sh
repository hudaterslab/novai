#!/bin/bash

# 현재 스크립트를 실행하는 사용자 이름 자동 획득 (fishduke 등)
ACTUAL_USER=$USER

echo "=========================================="
echo " 1. 바탕화면 자동 로그인 (GDM3) 자동 설정"
echo "=========================================="
# 만약을 위해 원본 설정 파일 백업
sudo cp /etc/gdm3/custom.conf /etc/gdm3/custom.conf.bak

# 파일 내에 AutomaticLogin 옵션이 있는지 확인 후 정규식(sed)으로 주석 해제 및 계정 삽입
if grep -iq "AutomaticLoginEnable" /etc/gdm3/custom.conf; then
    sudo sed -i -E 's/^#?\s*AutomaticLoginEnable\s*=\s*.*/AutomaticLoginEnable=True/i' /etc/gdm3/custom.conf
    sudo sed -i -E "s/^#?\s*AutomaticLogin\s*=\s*.*/AutomaticLogin=$ACTUAL_USER/i" /etc/gdm3/custom.conf
else
    # 옵션이 아예 없는 경우 [daemon] 섹션 바로 아래에 강제 추가
    sudo sed -i "/\[daemon\]/a AutomaticLoginEnable=True\nAutomaticLogin=$ACTUAL_USER" /etc/gdm3/custom.conf
fi
echo "✅ 사용자 [$ACTUAL_USER] 바탕화면 자동 로그인 세팅 완료!"

echo "=========================================="
echo " 2. 필수 패키지 설치 (Tailscale & RDP)"
echo "=========================================="
sudo apt update && sudo apt install -y curl gpg gnome-remote-desktop

# Tailscale 설치
curl -fsSL "https://pkgs.tailscale.com/stable/ubuntu/$(lsb_release -cs).noarmor.gpg" | sudo tee /usr/share/keyrings/tailscale.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/tailscale.gpg] https://pkgs.tailscale.com/stable/ubuntu $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/tailscale.list
sudo apt update && sudo apt install -y tailscale

echo "=========================================="
echo " 3. Ubuntu 원격 데스크톱 (RDP) 기본 설정"
echo "=========================================="
# SSH 환경에서도 GUI 데스크톱 세션과 통신할 수 있도록 환경변수 강제 주입
export XDG_RUNTIME_DIR=/run/user/$(id -u)
export DBUS_SESSION_BUS_ADDRESS="unix:path=${XDG_RUNTIME_DIR}/bus"

CERT_DIR="$HOME"
CRT_FILE="$CERT_DIR/rdp.crt"
KEY_FILE="$CERT_DIR/rdp.key"

rm -f "$CRT_FILE" "$KEY_FILE"
openssl req -x509 -newkey rsa:2048 -nodes -keyout "$KEY_FILE" -out "$CRT_FILE" -days 3650 -subj "/C=KR/O=Home/CN=ubuntu-rdp" 2>/dev/null

chmod 600 "$KEY_FILE"
chmod 644 "$CRT_FILE"

grdctl rdp set-tls-cert "$CRT_FILE"
grdctl rdp set-tls-key "$KEY_FILE"
grdctl rdp set-credentials asus 1234
gsettings set org.gnome.desktop.remote-desktop.rdp view-only false

echo "=========================================="
echo " 4. RDP 속도 최적화 및 절전/잠금 방지 설정"
echo "=========================================="
gsettings set org.gnome.desktop.session idle-delay 0
gsettings set org.gnome.desktop.screensaver lock-enabled false
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 'nothing'
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-battery-type 'nothing'
gsettings set org.gnome.desktop.interface enable-animations false
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target > /dev/null 2>&1

echo "=========================================="
echo " 5. RDP 서비스 활성화 및 재시작 적용"
echo "=========================================="
grdctl rdp enable
systemctl --user restart gnome-remote-desktop

echo "=========================================="
echo " 🎉 자동 셋팅 완료! (중요: 재부팅 필수)"
echo "=========================================="
echo " ⚠️ 새로 설정된 '자동 로그인'이 켜져서 암호 지갑(Keyring)이 풀리려면"
echo " 현재 상태에서 PC를 꼭 한 번 재부팅하셔야 합니다."
echo " 명령어: sudo reboot"
echo "=========================================="
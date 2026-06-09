import os
import gi

# 환경변수 청소 (multi_event.py와 동일한 조건 부여)
if "GST_PLUGIN_PATH" in os.environ:
    del os.environ["GST_PLUGIN_PATH"]
os.environ["GST_PLUGIN_SYSTEM_PATH"] = "/usr/lib/x86_64-linux-gnu/gstreamer-1.0"
os.environ["LIBVA_DRIVER_NAME"] = "iHD"
os.environ["GST_VAAPI_ALL_DRIVERS"] = "1"

gi.require_version('Gst', '1.0')
from gi.repository import Gst

Gst.init(None)

features_to_check = [
    "vaapih264dec", 
    "vaapih265dec", 
    "decodebin", 
    "uridecodebin",
    "dxinfer",       # 딥엑스 플러그인이 섞여 들어오는지 확인
    "h264parse"
]

print("=== 🔍 GStreamer Python Runtime Plugin Check ===")
for f_name in features_to_check:
    feature = Gst.Registry.get().lookup_feature(f_name)
    if feature:
        plugin = feature.get_plugin()
        print(f"[✅ FOUND] {f_name:12} -> Plugin: {plugin.get_name()}, Path: {plugin.get_filename()}")
    else:
        print(f"[❌ MISSING] {f_name:12} -> 파이썬이 이 플러그인을 찾지 못했습니다!")
print("================================================")

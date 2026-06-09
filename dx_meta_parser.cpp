#include <gst/gst.h>
#include <string.h>
#include "gst-dxframemeta.hpp"
#include "gst-dxobjectmeta.hpp"

extern "C" {
    // 100% 안전한 GstMeta 탐색 로직
    DXFrameMeta* find_dx_meta_safely(GstBuffer* buf) {
        if (!buf) return NULL;
        gpointer state = NULL;
        GstMeta* meta;
        while ((meta = gst_buffer_iterate_meta(buf, &state))) {
            if (meta->info && meta->info->api) {
                const gchar* api_name = g_type_name(meta->info->api);
                if (api_name && strstr(api_name, "DXFrameMeta") != NULL) {
                    return (DXFrameMeta*)meta;
                }
            }
        }
        return NULL;
    }

    // 파이썬으로 넘겨줄 1. 전체 객체 수 반환
    int get_num_objects(GstBuffer* buf) {
        DXFrameMeta* meta = find_dx_meta_safely(buf);
        if (!meta) return 0;
        return (int)meta->_object_meta_list.size();
    }

    // 파이썬으로 넘겨줄 2. 개별 객체 BBox 반환
    void get_object_data(GstBuffer* buf, int idx, float* x1, float* y1, float* x2, float* y2, float* conf, int* class_id) {
        DXFrameMeta* meta = find_dx_meta_safely(buf);
        if (!meta) return;

        int size = (int)meta->_object_meta_list.size();
        if (idx < 0 || idx >= size) return;
        
        DXObjectMeta* obj = meta->_object_meta_list[idx];
        if (!obj) return;
        
        *x1 = obj->_box[0];
        *y1 = obj->_box[1];
        *x2 = obj->_box[2];
        *y2 = obj->_box[3];
        *conf = obj->_confidence;
        *class_id = obj->_label;
    }
}
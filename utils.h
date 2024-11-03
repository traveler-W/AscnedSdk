#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>
#include <queue>
#include <unordered_map>
#include <cmath>
#include <dirent.h>
#include <string.h>
#include <vector>
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/types_c.h"
#include "opencv2/highgui/highgui.hpp"
#include "acllite_dvpp_lite/ImageProc.h"
#include "acllite_om_execute/ModelProc.h"
#include "acllite_media/CameraRead.h"
#include "acllite_dvpp_lite/VideoRead.h"
#include "acllite_common/Queue.h"
#include "label.h"
#include "xhhwvdec_api.h"
#include <cstring>
using namespace std;
using namespace acllite;
using namespace cv;
enum
{
    XHA_OK = 0,
    XHA_RROR = -1,
};
typedef struct FRAME_INFO_TS
{
    void *frame;
    unsigned int width;      // 实际宽
    unsigned int height;     // 实际高
    unsigned int vir_width;  // 虚宽
    unsigned int vir_height; // 虚高
    int h2645;
    int img_type;
} frame_info_hw;

namespace xha
{
    static ImageData ConvertFrameInfoToImageData(frame_info_hw *frameInfo)
    {
        uint32_t dataSize;
        if (frameInfo->frame != NULL)
        {
            dataSize = strlen((const char *)frameInfo->frame);
        }
        // 使用智能指针包装原始数据指针
        std::shared_ptr<uint8_t> data(static_cast<uint8_t *>(frameInfo->frame), [](uint8_t *) {});

        // 创建 ImageData 实例并返回
        return ImageData(data, dataSize, 1920, 1080, PIXEL_FORMAT_YUV_SEMIPLANAR_420);
    }

    static uint32_t cvWidth = 1920;
    static uint32_t cvHeight = 1080;
    static cv::Mat cvImg = cv::Mat(cvHeight, cvWidth, CV_8UC3, cv::Scalar(0, 0, 0));
    static uint32_t modelWidth = 640;
    static uint32_t modelHeight = 640;
    static aclrtContext context = nullptr;
    static int32_t deviceId = 0;
    // 传入数据载体
    struct MsgData
    {
        std::shared_ptr<uint8_t> data = nullptr;
        uint32_t size = 0;
        bool videoEnd = false;
        cv::Mat srcImg;
    };
    // 传出数据载体
    struct MsgOut
    {
        cv::Mat srcImg;
        bool videoEnd = false;
        vector<InferenceOutput> inferOutputs;
    };

    typedef struct BoundBox
    {
        float x;
        float y;
        float width;
        float height;
        float score;
        size_t classIndex;
        size_t index;
    } BoundBox;

    typedef struct result_
    {
        std::string class_name;
        float score;
        int label;

        int x1;
        int y1;
        int x2;
        int y2;
        int width;
        int height;

    } XhObjectInfo;

    typedef struct Ascend
    {
        std::string model_path; // 模型路径
        ModelProc modelProcess; // 模型加载和推理
        int isHost;             // 判断开发环境
        ImageProc imageProcess; // 视频帧处理器
        vector<XhObjectInfo> results;
        float confidenceThreshold = 0.25;
        size_t classNum = 3;
    } XHASess;

}
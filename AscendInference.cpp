#include "AscendInference.h"

int XHSDK_API XHAscendInit()
{
    std::cout << "\033[36m[XHAscendInit is ok]\n\033[0m" << std::endl;
    return XHA_OK;
}

int XHSDK_API XHAscendCreate(XHSDK_Sess *sess, std::string ModelPath, float confidenceThreshold, int classNum)
{
    if (!sess)
    {
        return XHA_RROR;
    }
    if (ModelPath == "")
    {
        std::cout << "ModelPath is null!" << std::endl;
        return XHA_RROR;
    }
    xha::XHASess *xha_sess = new xha::XHASess;
    xha_sess->model_path = ModelPath;
    xha_sess->modelProcess.Load(xha_sess->model_path);
    xha_sess->confidenceThreshold = confidenceThreshold;
    xha_sess->classNum = classNum;
    *sess = xha_sess;
    return XHA_OK;
}

int XHSDK_API XHAscendDetect(XHSDK_Sess sess, FRAME_INFO frame_data, xha::XhObjectInfo **object, int *count,int debug)
{
    if (!sess)
    {
        return XHA_RROR;
    }
    if (frame_data == nullptr)
    {
        std::cout << "frame is null!" << std::endl;
        return XHA_RROR;
    }
    frame_info_hw *frame_mid = (frame_info_hw *)frame_data;
    ImageData frame = xha::ConvertFrameInfoToImageData(frame_mid);
    if (debug)
    {
        std::cout << "fame_size:" << frame.size << " " << "frame_width:" << frame.width << " " << "frame_height:" << frame.height << " " << "frame_aliwidth:" << frame.alignWidth << " " << "frame_aliheight:" << frame.alignHeight << std::endl;
    }
    ImageProc imageProcess;
    ImageSize modelSize(xha::modelWidth, xha::modelHeight);
    xha::XHASess *xhasess = (xha::XHASess *)sess;
    xhasess->isHost = aclrtRunMode();
    ImageData dst;
    imageProcess.Resize(frame, dst, modelSize, RESIZE_PROPORTIONAL_UPPER_LEFT);
    if (dst.data == nullptr)
    {
        std::cout << "dst is null!" << std::endl;
        return XHA_RROR;
    }
    xha::MsgData msgData;
    msgData.data = dst.data;
    msgData.size = dst.size;
    msgData.videoEnd = false;
    cv::Mat yuyvImg(frame.height * 1.5, frame.width, CV_8UC1);
    if (xhasess->isHost)
    {
        // void* hostDataBuffer = CopyDataToHost(frame.data.get(), frame.size);
        // memcpy(yuyvImg.data, (unsigned char*)hostDataBuffer, frame.size);
        // FreeHostMem(hostDataBuffer);
        // hostDataBuffer = nullptr;
        memcpy(yuyvImg.data, (unsigned char *)frame.data.get(), frame.size);
    }
    else
    {
        memcpy(yuyvImg.data, (unsigned char *)frame.data.get(), frame.size);
    }
    cv::cvtColor(yuyvImg, msgData.srcImg, cv::COLOR_YUV2RGB_NV21);

    // 开始进行推理
    int ret = xhasess->modelProcess.CreateInput(static_cast<void *>(msgData.data.get()), msgData.size);
    CHECK_RET(ret, LOG_PRINT("[ERROR] Create model input failed."); return XHA_RROR);
    xha::MsgOut msgOut;
    msgOut.srcImg = msgData.srcImg;
    msgOut.videoEnd = msgData.videoEnd;
    xhasess->modelProcess.Execute(msgOut.inferOutputs);
    cv::Mat srcImage = msgOut.srcImg;
    // 后处理过程
    uint32_t outputDataBufId = 0;
    float *classBuff = static_cast<float *>(msgOut.inferOutputs[outputDataBufId].data.get());
    float confidenceThreshold = xhasess->confidenceThreshold;
    size_t classNum = xhasess->classNum;
    size_t offset = 5;
    size_t totalNumber = classNum + offset;
    size_t modelOutputBoxNum = 25200;
    size_t startIndex = 5;
    int srcWidth = srcImage.cols;
    int srcHeight = srcImage.rows;

    vector<xha::BoundBox> boxes;
    size_t yIndex = 1;
    size_t widthIndex = 2;
    size_t heightIndex = 3;
    size_t classConfidenceIndex = 4;
    float widthScale = (float)(srcWidth) / xha::modelWidth;
    float heightScale = (float)(srcHeight) / xha::modelHeight;
    float finalScale = (widthScale > heightScale) ? widthScale : heightScale;
    for (size_t i = 0; i < modelOutputBoxNum; ++i)
    {
        float maxValue = 0;
        float maxIndex = 0;
        for (size_t j = startIndex; j < totalNumber; ++j)
        {
            float value = classBuff[i * totalNumber + j] * classBuff[i * totalNumber + classConfidenceIndex];
            if (value > maxValue)
            {
                maxIndex = j - startIndex;
                maxValue = value;
            }
        }
        float classConfidence = classBuff[i * totalNumber + classConfidenceIndex];
        if (classConfidence >= confidenceThreshold)
        {
            size_t index = i * totalNumber + maxIndex + startIndex;
            float finalConfidence = classConfidence * classBuff[index];
            xha::BoundBox box;
            box.x = classBuff[i * totalNumber] * finalScale;
            box.y = classBuff[i * totalNumber + yIndex] * finalScale;
            box.width = classBuff[i * totalNumber + widthIndex] * finalScale;
            box.height = classBuff[i * totalNumber + heightIndex] * finalScale;
            box.score = finalConfidence;
            box.classIndex = maxIndex;
            box.index = i;
            if (maxIndex < classNum)
            {
                boxes.push_back(box);
            }
        }
    }
    vector<xha::BoundBox> result;
    result.clear();
    float NMSThreshold = 0.45;
    int32_t maxLength = xha::modelWidth > xha::modelHeight ? xha::modelWidth : xha::modelHeight;
    std::sort(boxes.begin(), boxes.end(), [](xha::BoundBox box1, xha::BoundBox box2)
              { return box1.score > box2.score; });
    xha::BoundBox boxMax;
    xha::BoundBox boxCompare;
    while (boxes.size() != 0)
    {
        size_t index = 1;
        result.push_back(boxes[0]);
        while (boxes.size() > index)
        {
            boxMax.score = boxes[0].score;
            boxMax.classIndex = boxes[0].classIndex;
            boxMax.index = boxes[0].index;
            boxMax.x = boxes[0].x + maxLength * boxes[0].classIndex;
            boxMax.y = boxes[0].y + maxLength * boxes[0].classIndex;
            boxMax.width = boxes[0].width;
            boxMax.height = boxes[0].height;

            boxCompare.score = boxes[index].score;
            boxCompare.classIndex = boxes[index].classIndex;
            boxCompare.index = boxes[index].index;
            boxCompare.x = boxes[index].x + boxes[index].classIndex * maxLength;
            boxCompare.y = boxes[index].y + boxes[index].classIndex * maxLength;
            boxCompare.width = boxes[index].width;
            boxCompare.height = boxes[index].height;
            float xLeft = max(boxMax.x, boxCompare.x);
            float yTop = max(boxMax.y, boxCompare.y);
            float xRight = min(boxMax.x + boxMax.width, boxCompare.x + boxCompare.width);
            float yBottom = min(boxMax.y + boxMax.height, boxCompare.y + boxCompare.height);
            float width = max(0.0f, xRight - xLeft);
            float hight = max(0.0f, yBottom - yTop);
            float area = width * hight;
            float iou = area / (boxMax.width * boxMax.height + boxCompare.width * boxCompare.height - area);
            if (iou > NMSThreshold)
            {
                boxes.erase(boxes.begin() + index);
                continue;
            }
            ++index;
        }
        boxes.erase(boxes.begin());
    }
    const double fountScale = 0.5;
    const uint32_t lineSolid = 2;
    const uint32_t labelOffset = 11;
    const cv::Scalar fountColor(0, 0, 255);
    const vector<cv::Scalar> colors{
        cv::Scalar(237, 149, 100), cv::Scalar(0, 215, 255),
        cv::Scalar(50, 205, 50), cv::Scalar(139, 85, 26)};

    int half = 2;
    std::vector<xha::XhObjectInfo> &results = xhasess->results;
    results.clear();
    for (size_t i = 0; i < result.size(); ++i)
    {
        if (result[i].score < 0.7)
        {
            continue;
        }

        cv::Point leftUpPoint, rightBottomPoint;
        leftUpPoint.x = result[i].x - result[i].width / half;
        leftUpPoint.y = result[i].y - result[i].height / half;
        rightBottomPoint.x = result[i].x + result[i].width / half;
        rightBottomPoint.y = result[i].y + result[i].height / half;
        // cv::rectangle(srcImage, leftUpPoint, rightBottomPoint, colors[i % colors.size()], lineSolid);
        string className = label2[result[i].classIndex];
        string markString = to_string(result[i].score) + ":" + className;

        xha::XhObjectInfo item;
        item.class_name = className;
        item.label = result[i].classIndex;
        item.score = result[i].score;
        item.x1 = leftUpPoint.x;
        item.y1 = leftUpPoint.y;
        item.x2 = rightBottomPoint.x;
        item.y2 = rightBottomPoint.y;
        item.width = rightBottomPoint.x - leftUpPoint.x;
        item.height = rightBottomPoint.y - leftUpPoint.y;

        results.push_back(item);

        // textPrint += markString;
         cv::putText(srcImage, markString, cv::Point(leftUpPoint.x, leftUpPoint.y + labelOffset),
                     cv::FONT_HERSHEY_COMPLEX, fountScale, fountColor);
        //  do you konw this is me and you
    }
    *object = results.data();
    *count = results.size();
    return XHA_OK;
}

int XHSDK_API XHAscendDestory(XHSDK_Sess *sess)
{
    if (!sess)
    {
        return XHA_RROR;
    }
    if (!*sess)
    {
        return XHA_RROR;
    }
    xha::XHASess *xhsess = (xha::XHASess *)sess;
    if (xhsess)
    {
        delete xhsess;
        xhsess = nullptr;
    }
    return XHA_OK;
}

int XHSDK_API XHAscendFinally()
{
    std::cout << "\033[36m[XHAscendFinally is ok]\n\033[0m" << std::endl;
    return XHA_OK;
}
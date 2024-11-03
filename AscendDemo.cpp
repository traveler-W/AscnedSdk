#include "AscendInference.h"
#include<iostream>
#include<fstream>
#include<string>


int main(int argv, char *args[])
{

    xhrvmppvdec_api api;
    int ret = XHHWVDInit("/root/wzc/EdgeAndRobotics/Samples/YOLOV5USBCamera/src/activation", "SparkAI_202412_axlic_JH16_boxMT_RX6HeMpLrTMk2fhJiQVjVXzlIR0izvha",16,25);
    if (ret != XHHWVD_OK)
    {
        std::cout << "XHHWVDInit error" << std::endl;
        return 0;
    }
    string videoPath = "rtsp://192.168.2.54:555/live";
    ret = api.XHHWVDCreate(2, videoPath.c_str(), "");
    if (ret != XHHWVD_OK)
    {
        std::cout << "XHHWVDCreate error" << std::endl;
        return 0;
    }
    ret = XHAscendInit();
    if (ret != XHA_OK)
    {
        std::cout << "XHAscendInit error" << std::endl;
        return 0;
    }
    XHSDK_Sess sess;
    ret = XHAscendCreate(&sess, "../model/best.om", 0.25, 3);
    if (ret != XHA_OK)
    {
        std::cout << "XHAscendCreate error" << std::endl;
        return 0;
    }
    FRAME_INFO frame;
    int width = 1920;
    int height = 1080;
    int vir_width = 1920;
    int vir_height = 1080;

    int cur=0;
    int index=1;
    while (1)
    {
        ret = api.XHHWVDGetFrame(&frame,&width,&height,&vir_width,&vir_height);

        unsigned char *jpeg_buf;
        int buf_size;
        int w,h;
        ret=api.XHHWVDHardDrawCvtJPEG(&frame,&jpeg_buf,&buf_size,&w,&h);
        if(ret!=XHHWVD_OK)
        {
            printf("XHHWVDHardDrawCvtJPEG err:%d\n",ret);
        } 
        std::string pathJpeg="./TestJpeg_";
        pathJpeg+=std::to_string(index);
        pathJpeg+="_";
        if(cur%20==0)pathJpeg+=std::to_string(cur);
        pathJpeg+=".jpg";
        
        std::ofstream file;
        file.open(pathJpeg,std::ios::binary);
        file.write((const char*)jpeg_buf, buf_size);
        file.close();

        printf("Frame to JPEG:%d,w:%d,h%d\n",buf_size,w,h);
        ret=api.XHHWVDHardFreeJPEG(&jpeg_buf,0);//出错ret不是正常，所以导致后面的ret，std::cout<<"cap read error......."<<std::endl;
        if(ret!=XHHWVD_OK)
        {
            printf("XHHWVDHardFreeJPEG err:%d\n",ret);
        }
        cur++;
        if (ret!=XHHWVD_OK)
        {
            xha::XhObjectInfo *results;
            int count=0;
            // 开始进行推理
            ret = XHAscendDetect(sess, frame, &results, &count,1);
            if (ret != XHA_OK)
            {
                std::cout << "XHAscendDetect error!" << std::endl;
                break;
            }
            std::cout << "result_size:" << count << std::endl;
            for (int i = 0; i < count; i++)
            {
                std::cout << "class:" << results[i].class_name << " score:" << results[i].score << std::endl;
                std::cout << "[" << results[i].x1 << "," << results[i].y1 << "," << results[i].width << "," << results[i].height << "]" << std::endl;
            }
            std::cout << "---------------------------------------------------------------------------------------------------" << std::endl
                      << std::endl;
        }
        else{
            std::cout<<"cap read error......."<<std::endl;
        }
    }
    ret = XHAscendDestory(&sess);
    if (ret != XHA_OK)
    {
        std::cout << "XHAscendDestory error!" << std::endl;
    }
    ret = XHAscendFinally();
    return 0;
}
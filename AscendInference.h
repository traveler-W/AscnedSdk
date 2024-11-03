#pragma once
#include "utils.h"
#define XHSDK_API

#ifdef __cplusplus
extern "C"
{
#endif
    typedef void *XHSDK_Sess;

    int XHSDK_API XHAscendInit();
	typedef int(XHSDK_API *PXHAscendInit)();

    int XHSDK_API XHAscendCreate(XHSDK_Sess *sess,std::string ModelPath,float confidenceThreshold,int classNum);
    typedef int(XHSDK_API *PXHAscendCreate)(XHSDK_Sess*,std::string,float,int);

    int XHSDK_API XHAscendDetect(XHSDK_Sess sess,FRAME_INFO frame,xha::XhObjectInfo **object,int *count,int debug);
    typedef int(XHSDK_API *PXHAscendDetect)(XHSDK_Sess,FRAME_INFO,xha::XhObjectInfo **,int);

    int XHSDK_API XHAscendDestory(XHSDK_Sess*sess);
    typedef int(XHSDK_API *PXHAscendDestory)(XHSDK_Sess*);

    int XHSDK_API XHAscendFinally();
    typedef int(XHSDK_API *PXHAscendFinally)();

#ifdef __cplusplus
}
#endif
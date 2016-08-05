//
//  main.cpp
//  C++_Sharpen_Filter
//
//  Created by PARK JAICHANG on 7/25/16.
//  Copyright © 2016 JAICHANGPARK. All rights reserved.
//

#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include <sys/time.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <CoreImage/CoreImage.h>
#include <emmintrin.h>
#include <immintrin.h>



unsigned int j = 0;
unsigned int i = 0;
//32bit
struct timeval start, stop ;
double result_time = 0;

double timedifference_msec(struct timeval t0, struct timeval t1);
bool Convolution3x3C(unsigned char *pSrc, unsigned char *pDest, unsigned int nImageWidth, unsigned int nImageHeight, unsigned int *ROIPoint, short *pKernel);
bool Convolution3x3Instrinsic(unsigned char *pSrc, unsigned char *pDest, unsigned int nImageWidth, unsigned int nImageHeight, unsigned int *ROIPoint, short *pKernel);
bool Convolution3x3Instrinsic2(unsigned char *pSrc, unsigned char *pDest, unsigned int nImageWidth, unsigned int nImageHeight, unsigned int *ROIPoint, short *pKernel);

int main(int argc, const char * argv[]) {
    //3078x1024
    //256x256
     const unsigned int WIDTH = 3078;
     const unsigned int HEIGHT = 1024;
     const unsigned int IMAGE_SIZE = WIDTH * HEIGHT;
     //const unsigned int BMP_HEADER_LENGTH = 1064;
     
    unsigned char *pSrc = new unsigned char[IMAGE_SIZE];
    unsigned char *pResult = new unsigned char[IMAGE_SIZE];
    
     unsigned int ROI[4] = {0,0,WIDTH,HEIGHT};
     short Kernel[9] = {-1,-1,-1,-1,9,-1,-1,-1,-1}; //sharpen
    
    gettimeofday(&start, NULL);
    Convolution3x3C(pSrc, pResult, WIDTH, HEIGHT, ROI, Kernel);
    gettimeofday(&stop, NULL);
    result_time = timedifference_msec(stop,start);
    printf("C: RESULT : code executed in %f microsecond \n",result_time);
    
    gettimeofday(&start, NULL);
    Convolution3x3Instrinsic(pSrc, pResult, WIDTH, HEIGHT, ROI, Kernel);
    gettimeofday(&stop, NULL);
    result_time = timedifference_msec(stop,start);
    printf("C_intrinsic : RESULT : code executed in %f microsecond \n",result_time);
    gettimeofday(&start, NULL);
    Convolution3x3Instrinsic2(pSrc, pResult, WIDTH, HEIGHT, ROI, Kernel);
    gettimeofday(&stop, NULL);
    result_time = timedifference_msec(stop,start);
    printf("C_intrinsic2 : RESULT : code executed in %f microsecond \n",result_time);
    return 0;
}

bool Convolution3x3C(unsigned char *pSrc, unsigned char *pDest, unsigned int nImageWidth, unsigned int nImageHeight, unsigned int *ROIPoint, short *pKernel){
    
    unsigned int nStartX = ROIPoint[0];
    unsigned int nStartY = ROIPoint[1];
    unsigned int nEndX =  ROIPoint[2];
    unsigned int nEndY = ROIPoint[3];
    
    if (0 == nStartX) {
        nStartX = 1;
    }
    if (0 == nStartY) {
        nStartY = 1;
    }
    if (nImageWidth == nEndX) {
        nEndX = nImageWidth - 1 ;
    }
    if (nImageHeight == nEndX) {
        nEndY = nImageHeight - 1;
    }
    
    short total = 0;
    short value[9] = {0};
    
    for (j = nStartY; j < nEndY; j++) {
        for (i = nStartX; i < nEndX; i++) {
            total = 0;
            
            value[0] = pSrc[i + j * nImageWidth - nImageHeight - 1];
            total += pKernel[0] * value[0];
            value[1] = pSrc[i + j * nImageWidth - nImageHeight];
            total += pKernel[1] * value[1];
            value[2] = pSrc[i + j * nImageWidth - nImageHeight + 1];
            total += pKernel[2] * value[2];
            value[3] = pSrc[i + j * nImageWidth - 1];
            total += pKernel[3] * value[3];
            value[4] = pSrc[i + j * nImageWidth];
            total += pKernel[4] * value[4];
            value[5] = pSrc[i + j * nImageWidth + 1];
            total += pKernel[5] * value[5];
            value[6] = pSrc[i + j * nImageWidth + nImageHeight - 1];
            total += pKernel[6] * value[6];
            value[7] = pSrc[i + j * nImageWidth + nImageHeight];
            total += pKernel[7] * value[7];
            value[8] = pSrc[i + j * nImageWidth + nImageHeight + 1];
            total += pKernel[8] * value[8];
            
            if (total < 0 ) {
                total = 0;
            }
            if (total > 255 ) {
                total = 255;
            }
            
            pDest[i+j*nImageWidth] = (unsigned char)total;
            
        }
    }
    
    
    return true;
    
}

bool Convolution3x3Instrinsic(unsigned char *pSrc, unsigned char *pDest, unsigned int nImageWidth, unsigned int nImageHeight, unsigned int *ROIPoint, short *pKernel)
{
    unsigned int nStartX = ROIPoint[0];
    unsigned int nStartY = ROIPoint[1];
    unsigned int nEndX = ROIPoint[2];
    unsigned int nEndY = ROIPoint[3];
    
    if(0 == nStartX) nStartX = 1;	
    if(0 == nStartY) nStartY = 1;	
    if(nImageWidth == nEndX) nEndX = nImageWidth - 1;
    if(nImageHeight == nEndY) nEndY = nImageHeight - 1;
    
    __m128i Kernel[9];
    
    int i = 0;
    
    for( i = 0; i < 9; i++)
        Kernel[i] = _mm_set1_epi16(pKernel[i]);
    
    __m128i ImageLow[9];
    __m128i ImageHigh[9];
  //  __m128i Dest;
    __m128i ZeroData = _mm_setzero_si128();
    
    __m128i ResultHigh;
    __m128i ResultLow;
    
    unsigned char * iterSrc;
    unsigned char * iterDest;
    unsigned int j = 0;
    
    for(j = nStartY; j < nEndY; j++)
    {
        for( i = nStartX; i < nEndX; i+=16)
        {
            iterSrc = pSrc+i+ nImageWidth*j;
            iterDest = pDest+i+ nImageWidth*j;
            
            ResultHigh = _mm_setzero_si128();
            ResultLow = _mm_setzero_si128();
            
            ImageLow[0] = _mm_loadu_si128((__m128i*)(iterSrc-1-nImageWidth));
            ImageLow[1] = _mm_loadu_si128((__m128i*)(iterSrc-nImageWidth));
            ImageLow[2] = _mm_loadu_si128((__m128i*)(iterSrc+1-nImageWidth));
            
            ImageLow[3] = _mm_loadu_si128((__m128i*)(iterSrc-1));
            ImageLow[4] = _mm_loadu_si128((__m128i*)(iterSrc));
            ImageLow[5] = _mm_loadu_si128((__m128i*)(iterSrc+1));
            
            ImageLow[6] = _mm_loadu_si128((__m128i*)(iterSrc-1+nImageWidth));
            ImageLow[7] = _mm_loadu_si128((__m128i*)(iterSrc+nImageWidth));
            ImageLow[8] = _mm_loadu_si128((__m128i*)(iterSrc+1+nImageWidth));
            
            for( int i = 0; i < 9 ; i++)
            {
                ImageHigh[i] = _mm_unpackhi_epi8(ImageLow[i],ZeroData);
                ImageLow[i] = _mm_unpacklo_epi8(ImageLow[i],ZeroData);
                
                ImageLow[i] = _mm_mullo_epi16(ImageLow[i],Kernel[i]);
                ImageHigh[i] = _mm_mullo_epi16(ImageHigh[i],Kernel[i]);
                
                ResultHigh = _mm_adds_epi16(ResultHigh,ImageHigh[i]);
                ResultLow = _mm_adds_epi16(ResultLow,ImageLow[i]);
            }
            
            ResultLow = _mm_packus_epi16(ResultLow,ResultHigh);
            
            _mm_storeu_si128( (__m128i*)iterDest,ResultLow);
            
        }
    }
    
    return true;
}

bool Convolution3x3Instrinsic2(unsigned char *pSrc, unsigned char *pDest, unsigned int nImageWidth, unsigned int nImageHeight, unsigned int *ROIPoint, short *pKernel)
{
    unsigned int nStartX = ROIPoint[0];
    unsigned int nStartY = ROIPoint[1];
    unsigned int nEndX = ROIPoint[2];
    unsigned int nEndY = ROIPoint[3];
    
    if(0 == nStartX) nStartX = 1;	//필터링 에러 처리 
    if(0 == nStartY) nStartY = 1;	//필터링 에러 처리 
    if(nImageWidth == nEndX) nEndX = nImageWidth - 1;
    if(nImageHeight == nEndY) nEndY = nImageHeight - 1;
    
    __m128i Kernel[9];
    
    int i = 0;
    unsigned int j = 0;
    
    for( i = 0; i < 9; i++)
        Kernel[i] = _mm_set1_epi16(pKernel[i]);
    
    __m128i ImageLow;
    __m128i ImageHigh;
   // __m128i Dest;
    __m128i ZeroData = _mm_setzero_si128();
    
    __m128i ResultHigh;
    __m128i ResultLow;
    
    unsigned char * iterSrc;
    unsigned char * iterDest;
    
    for( j = nStartY; j < nEndY; j++)
    {
        for( i = nStartX; i < nEndX; i+=16)
        {
            iterSrc = pSrc+i+ nImageWidth*j;
            iterDest = pDest+i+ nImageWidth*j;
            
            ResultHigh = _mm_setzero_si128();
            ResultLow = _mm_setzero_si128();
            
            ImageLow = _mm_loadu_si128((__m128i*)(iterSrc-1-nImageWidth));
            
            ImageHigh = _mm_unpackhi_epi8(ImageLow,ZeroData);
            ImageLow = _mm_unpacklo_epi8(ImageLow,ZeroData);
            
            ImageLow = _mm_mullo_epi16(ImageLow,Kernel[0]);
            ImageHigh = _mm_mullo_epi16(ImageHigh,Kernel[0]);
            
            ResultHigh = _mm_adds_epi16(ResultHigh,ImageHigh);
            ResultLow = _mm_adds_epi16(ResultLow,ImageLow);
            
            ImageLow = _mm_loadu_si128((__m128i*)(iterSrc-nImageWidth));
            
            ImageHigh = _mm_unpackhi_epi8(ImageLow,ZeroData);
            ImageLow = _mm_unpacklo_epi8(ImageLow,ZeroData);
            
            ImageLow = _mm_mullo_epi16(ImageLow,Kernel[1]);
            ImageHigh = _mm_mullo_epi16(ImageHigh,Kernel[1]);
            
            ResultHigh = _mm_adds_epi16(ResultHigh,ImageHigh);
            ResultLow = _mm_adds_epi16(ResultLow,ImageLow);
            
            ImageLow = _mm_loadu_si128((__m128i*)(iterSrc+1-nImageWidth));
            
            ImageHigh = _mm_unpackhi_epi8(ImageLow,ZeroData);
            ImageLow = _mm_unpacklo_epi8(ImageLow,ZeroData);
            
            ImageLow = _mm_mullo_epi16(ImageLow,Kernel[2]);
            ImageHigh = _mm_mullo_epi16(ImageHigh,Kernel[2]);
            
            ResultHigh = _mm_adds_epi16(ResultHigh,ImageHigh);
            ResultLow = _mm_adds_epi16(ResultLow,ImageLow);
            
            ImageLow = _mm_loadu_si128((__m128i*)(iterSrc-1));
            ImageHigh = _mm_unpackhi_epi8(ImageLow,ZeroData);
            ImageLow = _mm_unpacklo_epi8(ImageLow,ZeroData);
            
            ImageLow = _mm_mullo_epi16(ImageLow,Kernel[3]);
            ImageHigh = _mm_mullo_epi16(ImageHigh,Kernel[3]);
            
            ResultHigh = _mm_adds_epi16(ResultHigh,ImageHigh);
            ResultLow = _mm_adds_epi16(ResultLow,ImageLow);
            
            ImageLow = _mm_loadu_si128((__m128i*)(iterSrc));
            
            ImageHigh = _mm_unpackhi_epi8(ImageLow,ZeroData);
            ImageLow = _mm_unpacklo_epi8(ImageLow,ZeroData);
            
            
            ImageLow = _mm_mullo_epi16(ImageLow,Kernel[4]);
            ImageHigh = _mm_mullo_epi16(ImageHigh,Kernel[4]);
            
            ResultHigh = _mm_adds_epi16(ResultHigh,ImageHigh);
            ResultLow = _mm_adds_epi16(ResultLow,ImageLow);
            
            ImageLow = _mm_loadu_si128((__m128i*)(iterSrc+1));
            
            
            ImageHigh = _mm_unpackhi_epi8(ImageLow,ZeroData);
            ImageLow = _mm_unpacklo_epi8(ImageLow,ZeroData);
           
            
            ImageLow = _mm_mullo_epi16(ImageLow,Kernel[5]);
            ImageHigh = _mm_mullo_epi16(ImageHigh,Kernel[5]);
            
            ResultHigh = _mm_adds_epi16(ResultHigh,ImageHigh);
            ResultLow = _mm_adds_epi16(ResultLow,ImageLow);
            
            ImageLow = _mm_loadu_si128((__m128i*)(iterSrc-1+nImageWidth));
            
            
            ImageHigh = _mm_unpackhi_epi8(ImageLow,ZeroData);
            ImageLow = _mm_unpacklo_epi8(ImageLow,ZeroData);
            
            
            ImageLow = _mm_mullo_epi16(ImageLow,Kernel[6]);
            ImageHigh = _mm_mullo_epi16(ImageHigh,Kernel[6]);
            
            ResultHigh = _mm_adds_epi16(ResultHigh,ImageHigh);
            ResultLow = _mm_adds_epi16(ResultLow,ImageLow);
            
            ImageLow = _mm_loadu_si128((__m128i*)(iterSrc+nImageWidth));
          
            ImageHigh = _mm_unpackhi_epi8(ImageLow,ZeroData);
            ImageLow = _mm_unpacklo_epi8(ImageLow,ZeroData);
            
            
            ImageLow = _mm_mullo_epi16(ImageLow,Kernel[7]);
            ImageHigh = _mm_mullo_epi16(ImageHigh,Kernel[7]);
            
            ResultHigh = _mm_adds_epi16(ResultHigh,ImageHigh);
            ResultLow = _mm_adds_epi16(ResultLow,ImageLow);
            
            ImageLow = _mm_loadu_si128((__m128i*)(iterSrc+1+nImageWidth));
            ImageHigh = _mm_unpackhi_epi8(ImageLow,ZeroData);
            ImageLow = _mm_unpacklo_epi8(ImageLow,ZeroData);
            ImageLow = _mm_mullo_epi16(ImageLow,Kernel[8]);
            ImageHigh = _mm_mullo_epi16(ImageHigh,Kernel[8]);
            ResultHigh = _mm_adds_epi16(ResultHigh,ImageHigh);
            ResultLow = _mm_adds_epi16(ResultLow,ImageLow);

            ResultLow = _mm_packus_epi16(ResultLow,ResultHigh);
            
            _mm_storeu_si128( (__m128i*)iterDest,ResultLow);
            
        }
    }
    return true;
}


double timedifference_msec(struct timeval t0, struct timeval t1){
    
    return (double)(t0.tv_usec - t1.tv_usec) / 1000000 + (double)(t0.tv_sec - t1.tv_sec);
}

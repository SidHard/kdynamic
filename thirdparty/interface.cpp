/*
Copyright (c) 2011-2013 Gerhard Reitmayr, TU Graz

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

// This file contains different implementations to access the depth device
// The common API is defined in interface.h
// The returned depth buffers are mapped to the color buffer and store the
// depth at each pixel in mm. 0 marks an invalid pixel.

// This implementation uses the libfreenect library and pthreads for threading

#include <libfreenect/libfreenect.h>
#include <pthread.h>
#include <iostream>

using namespace std;

freenect_context *f_ctx;
freenect_device *f_dev;
bool gotDepth;
int depth_index;

pthread_t freenect_thread;
volatile bool die = false;

uint16_t * buffers[2];

void depth_cb(freenect_device *dev, void *v_depth, uint32_t timestamp)
{
    gotDepth = true;
    int next_buffer = (depth_index+1) % 2;
    freenect_set_depth_buffer(dev, buffers[depth_index]);
    depth_index = next_buffer;
}

void *freenect_threadfunc(void *arg)
{
    while(!die){
        int res = freenect_process_events(f_ctx);
        if (res < 0 && res != -10) {
            cout << "\nError "<< res << " received from libusb - aborting.\n";
            break;
        }
    }
    freenect_stop_depth(f_dev);
    freenect_stop_video(f_dev);
    freenect_close_device(f_dev);
    freenect_shutdown(f_ctx);
    return NULL;
}

int InitKinect( uint16_t * depth_buffer[2], unsigned char * rgb_buffer ){
    if (freenect_init(&f_ctx, NULL) < 0) {
        cout << "freenect_init() failed" << endl;
        return 1;
    }

    freenect_set_log_level(f_ctx, FREENECT_LOG_WARNING);
    freenect_select_subdevices(f_ctx, (freenect_device_flags)(FREENECT_DEVICE_MOTOR | FREENECT_DEVICE_CAMERA));

    int nr_devices = freenect_num_devices (f_ctx);
    if (nr_devices < 1)
        return 1;

    if (freenect_open_device(f_ctx, &f_dev, 0) < 0) {
        cout << "libfreenect: Could not open device" << endl;
        return 1;
    }

    gotDepth = false;
    depth_index = 0;
    buffers[0] = depth_buffer[0];
    buffers[1] = depth_buffer[1];
    
    freenect_set_depth_callback(f_dev, depth_cb);
    freenect_set_depth_mode(f_dev, freenect_find_depth_mode(FREENECT_RESOLUTION_MEDIUM, FREENECT_DEPTH_REGISTERED));
    freenect_set_depth_buffer(f_dev, buffers[depth_index]);

    freenect_set_video_mode(f_dev, freenect_find_video_mode(FREENECT_RESOLUTION_MEDIUM, FREENECT_VIDEO_RGB));
    freenect_set_video_buffer(f_dev, rgb_buffer);

    freenect_start_depth(f_dev);
    freenect_start_video(f_dev);

    int res = pthread_create(&freenect_thread, NULL, freenect_threadfunc, NULL);
    if(res){
        cout << "error starting kinect thread " << res << endl;
        return 1;
    }

    return 0;
}

void CloseKinect(){
    die = true;
    pthread_join(freenect_thread, NULL);
}

bool KinectFrameAvailable(){
    bool result = gotDepth;
    gotDepth = false;
    return result;
}

int GetKinectFrame(){
    return depth_index;
}

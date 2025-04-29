#ifndef STREAMING_SLICE_HPP
#define STREAMING_SLICE_HPP

#include <hls_stream.h>
#include <ap_int.h>

// StreamingSlice: reads input stream, outputs only selected elements
template<
    typename T,
    unsigned int NumIn,
    unsigned int NumOut,
    unsigned int StartIdx,
    unsigned int Step
>
void StreamingSlice(hls::stream<T> &in0, hls::stream<T> &out) {
#pragma HLS INLINE off
    T temp;
    unsigned int idx = 0;
    unsigned int out_idx = 0;
SliceLoop:
    for (idx = 0; idx < NumIn; idx++) {
#pragma HLS PIPELINE II=1
        temp = in0.read();
        if ((idx >= StartIdx) && (((idx - StartIdx) % Step) == 0) && (out_idx < NumOut)) {
            out.write(temp);
            out_idx++;
        }
    }
}

#endif

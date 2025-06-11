#include <hls_stream.h>
#include <ap_int.h>
#include <stdio.h> 


template<
    typename T,              // ap_int<24>
    unsigned int NumH_In,    // 65 (Size of axis dimension)
    unsigned int NumChannels,// 8  (Size of inner dimension)
    unsigned int NumH_Out,   // 17 (Slice length along axis)
    unsigned int StartH_Idx, // 0
    unsigned int StepH       // 4
>
void StreamingSlice(hls::stream<T> &in0, hls::stream<T> &out) {
    #pragma HLS INLINE off
    T temp;
    unsigned int h_input_idx = 0;    // Current H-slice 
    unsigned int h_output_count = 0; // H-slices outputted
    unsigned int total_elements_read = 0;
    const unsigned int TOTAL_INPUT_ELEMENTS = NumH_In * NumChannels;

    // Loop over all conceptual H-slices in the input
    ReadInputHLoop:
    for (h_input_idx = 0; h_input_idx < NumH_In; h_input_idx++) {
        bool keep_this_h_slice = (h_input_idx >= StartH_Idx) &&
                                 (((h_input_idx - StartH_Idx) % StepH) == 0) &&
                                 (h_output_count < NumH_Out);
        // Read all channels
        ReadChannelLoop:
        for (unsigned int c_idx = 0; c_idx < NumChannels; c_idx++) {
        #pragma HLS PIPELINE II=1
            if (total_elements_read >= TOTAL_INPUT_ELEMENTS) break; 

            temp = in0.read();
            total_elements_read++;

            if (keep_this_h_slice) {
                out.write(temp); 
            }
        } 

        if (keep_this_h_slice) {
            h_output_count++; 
        }
    } 
}
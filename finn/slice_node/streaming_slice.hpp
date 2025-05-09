#include <hls_stream.h>
#include <ap_int.h>
#include <stdio.h> // Include for temporary debugging

// Add NumChannels as a template parameter
template<
    typename T,          // ap_int<24>
    unsigned int NumH_In,    // 65 (Size of axis dimension)
    unsigned int NumChannels, // 8 (Size of inner dimension)
    unsigned int NumH_Out,   // 17 (Slice length along axis)
    unsigned int StartH_Idx, // 0
    unsigned int StepH       // 4
>
void StreamingSlice(hls::stream<T> &in0, hls::stream<T> &out) {
    #pragma HLS INLINE off
    T temp;
    unsigned int h_input_idx = 0;       // Current H-slice index from input
    unsigned int h_output_count = 0;    // How many H-slices we've outputted
    unsigned int total_elements_read = 0;
    const unsigned int TOTAL_INPUT_ELEMENTS = NumH_In * NumChannels; // Total elements in the stream segment

    // Loop over all conceptual H-slices in the input
    ReadInputHLoop:
    for (h_input_idx = 0; h_input_idx < NumH_In; h_input_idx++) {

        // Determine if we should keep this H-slice based on H index and step
        bool keep_this_h_slice = (h_input_idx >= StartH_Idx) &&
                                 (((h_input_idx - StartH_Idx) % StepH) == 0) &&
                                 (h_output_count < NumH_Out);

        // Read all channels for this H-slice from the stream
        ReadChannelLoop:
        for (unsigned int c_idx = 0; c_idx < NumChannels; c_idx++) {
        // Ensure the pipeline pragma applies to the innermost loop for throughput
        #pragma HLS PIPELINE II=1

            // Prevent reading beyond the total expected stream elements if NumIn wasn't total
             if (total_elements_read >= TOTAL_INPUT_ELEMENTS) break; // Or handle error

            // ALWAYS read from input to consume the data
            temp = in0.read();
            total_elements_read++;

             // --- DEBUG PRINT ---
             if (h_input_idx < 2 || h_input_idx == 4) { // Print for H=0, H=1, H=4
                 long long temp_ll = temp; // Cast for printf
                 printf("HLS Corrected: h_in=%u, c_in=%u, read_temp=%lld (0x%llX), keep_h=%d\n",
                        h_input_idx, c_idx, temp_ll, temp_ll, keep_this_h_slice);
             }
             // --- END DEBUG ---

            if (keep_this_h_slice) {
                // --- DEBUG PRINT ---
                 if (h_input_idx < 2 || h_input_idx == 4) {
                     long long temp_ll = temp;
                     printf("HLS Corrected: h_in=%u, c_in=%u, writing_temp=%lld (0x%llX), h_out_cnt=%u\n",
                            h_input_idx, c_idx, temp_ll, temp_ll, h_output_count);
                 }
                // --- END DEBUG ---
                out.write(temp); // Write if H-slice is kept
            }
        } // End channel loop

        if (keep_this_h_slice) {
            h_output_count++; // Increment count AFTER processing all channels for the kept H-slice
        }
        
        // Optional: Break outer loop early if all outputs generated AND input fully consumed?
        // Needs care to ensure all input required is consumed even if not outputted.
        // The current structure ensures all NumH_In * NumChannels are read.

    } // End H loop
}

from driver import FINNExampleOverlay
from qonnx.core.datatype import DataType

# ——— PLATFORM & I/O DESCRIPTION ———
platform = "zynq-iodma"

io_shape_dict = {
    # FINN DataType for input and output tensors
    "idt" : [DataType['INT8']],
    "odt" : [DataType['INT17']],
    # shapes for input and output tensors (NHWC layout)
    "ishape_normal" : [(1, 1000, 1, 1)],
    "oshape_normal" : [(1, 1, 1, 5)],
    # folded / packed shapes below depend on idt/odt and input/output
    # PE/SIMD parallelization settings -- these are calculated by the
    # FINN compiler.
    "ishape_folded" : [(1, 1000, 1, 1, 1)],
    "oshape_folded" : [(1, 1, 1, 5, 1)],
    "ishape_packed" : [(1, 1000, 1, 1, 1)],
    "oshape_packed" : [(1, 1, 1, 5, 3)],
    "input_dma_name" : ['idma0'],
    "output_dma_name" : ['odma0'],
    "number_of_external_weights": 0,
    "num_inputs" : 1,
    "num_outputs" : 1,
}
print("⏳  Downloading bitstream & setting up FPGA driver…")
ol = FINNExampleOverlay(
    bitfile_name="resizer.bit",
    platform=platform,
    io_shape_dict=io_shape_dict,
    download=True      # first time you run it, you do want to flash
)
print("✅  FPGA ready.")
print(ol.ip_dict.keys())
# you can export `ol` so inference script can just import it:
__all__ = ["ol"]

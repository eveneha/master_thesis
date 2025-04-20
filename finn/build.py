import finn.builder.build_dataflow_config as build_cfg
import finn.builder.build_dataflow as build

# Configuration tailored for PYNQ-Z1
cfg = build_cfg.DataflowBuildConfig(
    output_dir="output_dir",
    fpga_part="xc7z020clg400-1",            # PYNQ-Z1 FPGA part
    platform="zynq-iodma",
    vivado_pynq_board="Pynq-Z1",
    shell_flow_type=build_cfg.ShellFlowType.VIVADO_ZYNQ,
    synth_clk_period_ns=10,                 # (100 MHz)
    generate_outputs=[
        build_cfg.DataflowOutputType.BITFILE,
        build_cfg.DataflowOutputType.PYNQ_DRIVER
    ],
    save_intermediate_models=True,
    verify_steps=[],                     # Faster, set True for debugging
)

# Run the build pipeline
onnx_model_path = "/home/eveneiha/finn/workspace/finn/tcn_v31.onnx"
build.build_dataflow_cfg(onnx_model_path, cfg)


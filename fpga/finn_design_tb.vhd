----------------------------------------------------------------------------------
-- Company:
-- Engineer:
--
-- Create Date: 05/04/2025 03:24:28 PM (Updated for 1000 inputs, single driver for sim_done_s)
-- Design Name:
-- Module Name: finn_stitch_tb - Behavioral
-- Project Name:
-- Target Devices:
-- Tool Versions:
-- Description: Testbench for FINN Stitched Dataflow Block Design
--
-- Dependencies: tb_input_data.txt, tb_expected_output.txt
--
-- Revision:
-- Revision 0.08 - Resolved multiply driven sim_done_s, 1000 inputs, I/O driven completion
----------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;
library std;
use std.textio.all;

entity finn_design_tb is -- Name from your original code
end finn_design_tb;

architecture behavior of finn_design_tb is

    -- Constants
    constant CLK_PERIOD         : time    := 10 ns;

    constant AXI_INPUT_WIDTH    : integer := 8;
    constant AXI_OUTPUT_WIDTH   : integer := 24;

    -- === Stream Length Calculation (FOR 1000 INPUTS) ===
    constant NUM_INPUT_SPATIAL_ELEMENTS : integer := 1000;
    constant INPUT_CHANNELS             : integer := 1;
    constant PE_IN                      : integer := 1; -- <<< VERIFY
    constant INPUT_SIMD                 : integer := INPUT_CHANNELS / PE_IN;
    constant NUM_INPUT_WORDS            : integer := NUM_INPUT_SPATIAL_ELEMENTS * INPUT_SIMD; -- Should be 1000 if PE_IN=1

    constant NUM_OUTPUT_SPATIAL_ELEMENTS: integer := 1;
    constant OUTPUT_CHANNELS            : integer := 5;
    constant PE_OUT                     : integer := 1; -- <<< VERIFY
    constant OUTPUT_SIMD                : integer := OUTPUT_CHANNELS / PE_OUT;
    constant NUM_OUTPUT_WORDS           : integer := NUM_OUTPUT_SPATIAL_ELEMENTS * OUTPUT_SIMD+45; -- Should be 5 if PE_OUT=1

    -- Input/Output Files
    constant INPUT_DATA_FILE    : string := "tb_input_data.txt";
    constant EXPECTED_DATA_FILE : string := "tb_expected_output.txt";
    constant OUTPUT_DATA_FILE   : string := "tb_hw_output.txt";

    constant UNCOMPARABLE_INT_VAL : integer := integer'high -1;

    -- Signals
    signal clk                  : std_logic := '0';
    signal rst_n                : std_logic := '0';
    signal sim_done_s           : boolean := false; -- Master sim done signal (driven ONLY by sim_control_proc)
    shared variable errors_v    : integer := 0;

    signal stimulus_finished_s  : boolean := false;
    signal monitor_finished_s   : boolean := false;

    -- New signals for error indication from I/O processes
    signal stimulus_fatal_error_s : boolean := false;
    signal monitor_fatal_error_s  : boolean := false;

    -- DUT Connections
    signal s_axis_tdata_dut     : std_logic_vector(AXI_INPUT_WIDTH-1 downto 0) := (others => '0');
    signal s_axis_tvalid_dut    : std_logic := '0';
    signal s_axis_tready_dut    : std_logic;
    signal m_axis_tdata_dut     : std_logic_vector(AXI_OUTPUT_WIDTH-1 downto 0);
    signal m_axis_tvalid_dut    : std_logic;
    signal m_axis_tready_dut    : std_logic := '0';

    component StreamingDataflowPartition_1_wrapper
        port (
            ap_clk          : in std_logic;
            ap_rst_n        : in std_logic;
            s_axis_0_tdata  : in std_logic_vector(AXI_INPUT_WIDTH-1 downto 0);
            s_axis_0_tvalid : in std_logic;
            s_axis_0_tready : out std_logic;
            m_axis_0_tdata  : out std_logic_vector(AXI_OUTPUT_WIDTH-1 downto 0);
            m_axis_0_tvalid : out std_logic;
            m_axis_0_tready : in std_logic
        );
    end component;

begin



    DUT_inst : StreamingDataflowPartition_1_wrapper
        port map (
            ap_clk        => clk,
            ap_rst_n      => rst_n,
            s_axis_0_tdata  => s_axis_tdata_dut,
            s_axis_0_tvalid => s_axis_tvalid_dut,
            s_axis_0_tready => s_axis_tready_dut,
            m_axis_0_tdata  => m_axis_tdata_dut,
            m_axis_0_tvalid => m_axis_tvalid_dut,
            m_axis_0_tready => m_axis_tready_dut
        );

    clk_gen : process
        variable clk_cycles : natural := 0;
    begin
        report "TB: clk_gen process started.";
        while not sim_done_s loop
            clk <= '0';
            wait for CLK_PERIOD / 2;
            clk <= '1';
            wait for CLK_PERIOD / 2;
            clk_cycles := clk_cycles + 1;
            if clk_cycles mod 200000 = 0 then
                report "TB: clk_gen heartbeat - cycle " & natural'image(clk_cycles) & ", Time: " & time'image(now);
            end if;
        end loop;
        report "TB: clk_gen process finished (sim_done_s is true). Total cycles generated: " & natural'image(clk_cycles);
        wait;
    end process clk_gen;

    reset_gen : process
    begin
        report "TB: reset_gen process started.";
        rst_n <= '0';
        report "TB: rst_n asserted (0).";
        wait for CLK_PERIOD * 5;
        rst_n <= '1';
        report "TB: rst_n de-asserted (1).";
        wait;
    end process reset_gen;

    stimulus_proc : process
        file infile           : text;
        variable infile_status  : file_open_status;
        variable inline         : line;
        variable file_value     : integer;
        variable packed_word    : std_logic_vector(AXI_INPUT_WIDTH-1 downto 0);
        variable words_sent     : integer := 0;
    begin
        report "Stimulus: Process Started. Expecting " & integer'image(NUM_INPUT_WORDS) & " words from " & INPUT_DATA_FILE & ".";
        stimulus_finished_s <= false;
        stimulus_fatal_error_s <= false; -- Initialize

        file_open(infile_status, infile, INPUT_DATA_FILE, read_mode);
        if infile_status /= open_ok then
            report "Stimulus FATAL: Could not open input file: " & INPUT_DATA_FILE severity failure;
            stimulus_fatal_error_s <= true; -- Signal fatal error
            errors_v := errors_v + 1;     -- Increment general error for file issue
        else
            s_axis_tvalid_dut <= '0';
            s_axis_tdata_dut  <= (others => '0');
            wait until rst_n = '1' or sim_done_s = true; -- Check sim_done_s (read-only)

            if sim_done_s then
                 report "Stimulus: Exiting early due to sim_done_s before reset release." severity warning;
            elsif stimulus_fatal_error_s then
                 report "Stimulus: Exiting early due to pre-loop fatal error." severity warning;
            else
                wait for CLK_PERIOD;
                report "Stimulus: Reset released, starting main loop.";
                main_stim_loop: for i in 0 to NUM_INPUT_WORDS - 1 loop
                    if sim_done_s or stimulus_fatal_error_s then -- Check sim_done_s or if an error already occurred
                        if sim_done_s then report "Stimulus: sim_done_s detected, exiting main_stim_loop at i=" & integer'image(i); end if;
                        if stimulus_fatal_error_s then report "Stimulus: fatal_error detected, exiting main_stim_loop at i=" & integer'image(i); end if;
                        exit main_stim_loop;
                    end if;

                    if endfile(infile) then
                        report "Stimulus Error: End of input file. Expected " & integer'image(NUM_INPUT_WORDS) & ", read " & integer'image(words_sent) & "." severity error;
                        errors_v := errors_v + (NUM_INPUT_WORDS - words_sent);
                        exit main_stim_loop; -- Or stimulus_fatal_error_s <= true; if this should stop all
                    end if;

                    readline(infile, inline);
                    if inline'length > 0 then
                        read(inline, file_value);
                        packed_word := std_logic_vector(to_signed(file_value, AXI_INPUT_WIDTH));
                        s_axis_tdata_dut  <= packed_word;
                        s_axis_tvalid_dut <= '1';

                        handshake_loop: loop
                            wait until rising_edge(clk) or sim_done_s = true;
                            if sim_done_s then exit handshake_loop; end if;
                            exit handshake_loop when s_axis_tready_dut = '1';
                        end loop handshake_loop;

                        if sim_done_s then s_axis_tvalid_dut <= '0'; exit main_stim_loop; end if;

                        if s_axis_tready_dut = '1' then
                            words_sent := words_sent + 1;
                        else
                            s_axis_tvalid_dut <= '0';
                            report "Stimulus: Handshake failed for word " & integer'image(i) & " (TREADY not high, sim_done_s is false)" severity warning;
                            exit main_stim_loop; -- Or stimulus_fatal_error_s <= true;
                        end if;
                    else
                        report "Stimulus Warning: Skipped empty line in input file for word " & integer'image(i) severity warning;
                    end if;
                end loop main_stim_loop;

                s_axis_tvalid_dut <= '0';
                s_axis_tdata_dut  <= (others => '0');

                if words_sent /= NUM_INPUT_WORDS and not sim_done_s and not stimulus_fatal_error_s then
                    report "Stimulus Warning: Only sent " & integer'image(words_sent) & " words, expected " & integer'image(NUM_INPUT_WORDS) severity warning;
                elsif not sim_done_s and not stimulus_fatal_error_s then
                    report "Stimulus: Successfully sent " & integer'image(words_sent) & " words.";
                end if;
            end if;
            if infile_status = open_ok then file_close(infile); end if;
        end if;

        if stimulus_fatal_error_s then
            report "Stimulus: Process finishing due to fatal file error.";
        else
            report "Stimulus: Process Finished after sending " & integer'image(words_sent) & " words. Current time: " & time'image(now);
        end if;
        stimulus_finished_s <= true; -- Always signal finished
        wait;
    end process stimulus_proc;

    monitor_proc : process
        file outfile            : text;
        variable outfile_status : file_open_status;
        file expectedfile       : text;
        variable expfile_status : file_open_status;
        variable outline        : line;
        variable expectedline   : line;
        variable file_value     : integer;
        variable hw_value_signed: signed(AXI_OUTPUT_WIDTH-1 downto 0);
        variable expected_val_signed: signed(AXI_OUTPUT_WIDTH-1 downto 0);
        variable hw_value_int   : integer;
        variable expected_val_int: integer;
        variable words_received : integer := 0;
        variable current_errors_monitor : integer := 0;
        variable monitor_wait_cycles : natural := 0;
    begin
        report "Monitor: Process Started. Expecting " & integer'image(NUM_OUTPUT_WORDS) & " words.";
        monitor_finished_s <= false;
        monitor_fatal_error_s <= false; -- Initialize

        file_open(outfile_status, outfile, OUTPUT_DATA_FILE, write_mode);
        if outfile_status /= open_ok then
            report "Monitor FATAL: Could not open output file: " & OUTPUT_DATA_FILE severity failure;
            monitor_fatal_error_s <= true;
            errors_v := errors_v + 1;
        end if;

        file_open(expfile_status, expectedfile, EXPECTED_DATA_FILE, read_mode);
        if expfile_status /= open_ok then
            report "Monitor FATAL: Could not open expected data file: " & EXPECTED_DATA_FILE severity failure;
            monitor_fatal_error_s <= true;
            errors_v := errors_v + 1;
        end if;

        m_axis_tready_dut <= '0';

        if not monitor_fatal_error_s then
            wait until rst_n = '1' or sim_done_s = true;
            if sim_done_s then
                 report "Monitor: Exiting early due to sim_done_s before reset release." severity warning;
            elsif monitor_fatal_error_s then
                 report "Monitor: Exiting early due to pre-loop fatal file error." severity warning;
            else
                m_axis_tready_dut <= '1';
                report "Monitor: Reset released, TREADY asserted, starting main loop.";

                monitor_loop : loop
                    if sim_done_s or monitor_fatal_error_s or words_received >= NUM_OUTPUT_WORDS then
                        exit monitor_loop;
                    end if;

                    monitor_wait_cycles := 0;
                    wait_tvalid_loop: loop
                        exit wait_tvalid_loop when m_axis_tvalid_dut = '1' or sim_done_s = true or monitor_fatal_error_s = true or words_received >= NUM_OUTPUT_WORDS;
                        wait for CLK_PERIOD;
                        monitor_wait_cycles := monitor_wait_cycles + 1;
                        if monitor_wait_cycles mod 100000 = 0 then
                            report "Monitor: Still waiting for TVALID on word " & integer'image(words_received) &
                                   ". Waited " & natural'image(monitor_wait_cycles) & " cycles for this word. m_axis_tvalid_dut=" & std_logic'image(m_axis_tvalid_dut) &
                                   ". Time: " & time'image(now);
                        end if;
                    end loop wait_tvalid_loop;

                    if sim_done_s or monitor_fatal_error_s or words_received >= NUM_OUTPUT_WORDS then exit monitor_loop; end if;

                    if m_axis_tvalid_dut = '1' then
                        wait until rising_edge(clk) or sim_done_s = true or monitor_fatal_error_s = true;
                        if sim_done_s or monitor_fatal_error_s then exit monitor_loop; end if;

                        hw_value_signed := signed(m_axis_tdata_dut);
                        hw_value_int := to_integer(hw_value_signed);
                        write(outline, hw_value_int);
                        writeline(outfile, outline);
                        report "Monitor: Rx word " & integer'image(words_received) & " value " & integer'image(hw_value_int) & " at time " & time'image(now);


                        expected_val_int := UNCOMPARABLE_INT_VAL;
                        if not endfile(expectedfile) then
                            readline(expectedfile, expectedline);
                            if expectedline'length > 0 then
                                read(expectedline, file_value);
                                expected_val_signed := to_signed(file_value, AXI_OUTPUT_WIDTH);
                                expected_val_int := to_integer(expected_val_signed);
                            else
                                report "Monitor Warning: Skipped empty line in expected file for output word " & integer'image(words_received) & "." severity warning;
                            end if;
                        else
                            report "Monitor Error: Ran out of expected data! Word " & integer'image(words_received) severity error;
                            current_errors_monitor := current_errors_monitor + 1;
                        end if;

                        if expected_val_int /= UNCOMPARABLE_INT_VAL and hw_value_signed /= expected_val_signed then
                            report "Monitor Error: Mismatch at output word " & integer'image(words_received) &
                                   ". Expected=" & integer'image(expected_val_int) &
                                   ", Got=" & integer'image(hw_value_int) severity warning;
                            current_errors_monitor := current_errors_monitor + 1;
                        end if;
                        words_received := words_received + 1;
                    end if;
                end loop monitor_loop;

                m_axis_tready_dut <= '0';

                if words_received < NUM_OUTPUT_WORDS and not sim_done_s and not monitor_fatal_error_s then
                    report "Monitor Warning: Received only " & integer'image(words_received) & " words, expected " & integer'image(NUM_OUTPUT_WORDS) severity warning;
                    current_errors_monitor := current_errors_monitor + (NUM_OUTPUT_WORDS - words_received);
                elsif not sim_done_s and not monitor_fatal_error_s then
                    report "Monitor: Successfully received " & integer'image(words_received) & " words.";
                end if;
            end if;
            if outfile_status = open_ok then file_close(outfile); end if;
            if expfile_status = open_ok then file_close(expectedfile); end if;
        end if;

        errors_v := errors_v + current_errors_monitor;
        if monitor_fatal_error_s then
            report "Monitor: Process finishing due to fatal file error.";
        else
            report "Monitor: Process Finished. Received " & integer'image(words_received) & " words. Errors by monitor: " & integer'image(current_errors_monitor) & ". Time: " & time'image(now);
        end if;
        monitor_finished_s <= true; -- Always signal finished
        wait;
    end process monitor_proc;

    sim_control_proc : process
        constant ONE_INFERENCE_CYCLES_ESTIMATE : integer := 1500000; -- Estimate from FPGA runtime
        -- Adjust if 1000 inputs means multiple inferences or a significantly different single inference time
        constant NUM_EXPECTED_INFERENCES_IN_TB : integer := 1; -- Assuming the 1000 inputs are for one overall DUT operation
        constant SAFETY_TIMEOUT_CYCLES         : integer := (ONE_INFERENCE_CYCLES_ESTIMATE * NUM_EXPECTED_INFERENCES_IN_TB) + 1000000; -- e.g., 1.5M + 1M = 2.5M cycles
        variable current_cycle                 : integer := 0;
    begin
        report "SimControl: Process Started. Safety timeout set to approx " & integer'image(SAFETY_TIMEOUT_CYCLES) & " cycles.";
        sim_done_s <= false; -- Initialize (this is the only driver for assignments to sim_done_s)

        control_loop: loop
            wait for CLK_PERIOD;
            current_cycle := current_cycle + 1;

            if stimulus_fatal_error_s or monitor_fatal_error_s then
                report "SimControl: Fatal error detected from I/O process. Ending simulation. Cycle: " & integer'image(current_cycle) & ", Time: " & time'image(now);
                exit control_loop;
            end if;

            if stimulus_finished_s and monitor_finished_s then
                report "SimControl: Both stimulus and monitor finished, exiting control_loop. Cycle: " & integer'image(current_cycle) & ", Time: " & time'image(now);
                exit control_loop;
            end if;

            if current_cycle >= SAFETY_TIMEOUT_CYCLES then
                report "SimControl: SAFETY TIMEOUT cycles reached (" & integer'image(current_cycle) & "), exiting control_loop. Time: " & time'image(now);
                exit control_loop;
            end if;
        end loop control_loop;

        -- Determine final status
        if stimulus_fatal_error_s or monitor_fatal_error_s then
            report "SimControl: Simulation ended due to FATAL I/O error." severity error;
            -- errors_v already incremented by I/O process if it was a file error
        elsif current_cycle >= SAFETY_TIMEOUT_CYCLES and not (stimulus_finished_s and monitor_finished_s) then
             report "SimControl: SAFETY TIMEOUT after " & integer'image(current_cycle) & " cycles! Stimulus_finished: " & boolean'image(stimulus_finished_s) & ", Monitor_finished: " & boolean'image(monitor_finished_s) severity error;
             errors_v := errors_v + 1;
        elsif not (stimulus_finished_s and monitor_finished_s) then
             -- This case might happen if fatal_error flags caused early exit but finished flags not yet set.
             report "SimControl: Simulation ended (likely due to fatal error or unexpected state). Stimulus_finished: " & boolean'image(stimulus_finished_s) & ", Monitor_finished: " & boolean'image(monitor_finished_s) & ". Cycle: " & integer'image(current_cycle) severity warning;
        else
             report "SimControl: Simulation completed based on I/O flags. Total cycles: " & integer'image(current_cycle) severity note;
        end if;

        sim_done_s <= true; -- <<<< THE ONLY PLACE sim_done_s IS DRIVEN TO TRUE

        report "SimControl: Waiting for final settles before summary.";
        wait for CLK_PERIOD * 10;

        report "----------------------------------------------------";
        report "SIMULATION SUMMARY (from SimControl):";
        if errors_v = 0 then
            report "Test PASSED. Total Errors: 0" severity note;
        else
            report "Test FAILED. Total Errors: " & integer'image(errors_v) severity error;
        end if;
        report "----------------------------------------------------";
        report "SimControl: Ending Simulation at time " & time'image(now);
        wait;
    end process sim_control_proc;

end architecture behavior;
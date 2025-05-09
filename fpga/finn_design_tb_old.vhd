----------------------------------------------------------------------------------
-- Company:
-- Engineer:
--
-- Create Date: 05/04/2025 03:24:28 PM
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
-- Revision 0.01 - File Created
-- Revision 0.02 - Added correct signed conversion, updated constants
-- Revision 0.03 - Updated simulation end condition logic
-- Additional Comments: Ensure constants match DUT wrapper and data files are correct.
----------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all; -- For ceil, log2
library std;
use std.textio.all;

entity finn_design_tb is
end finn_design_tb;

architecture behavior of finn_design_tb is

    -- Constants
    constant CLK_PERIOD         : time    := 10 ns; -- Matches target_clk_ns

    -- === Physical AXI Interface Widths (from Wrapper) ===
    constant AXI_INPUT_WIDTH    : integer := 8;  -- From s_axis_0_tdata [7:0]
    constant AXI_OUTPUT_WIDTH   : integer := 24; -- From m_axis_0_tdata [23:0]

    -- === Logical Data Parameters (for calculation & packing) ===
    constant INPUT_ELEMENT_BW   : integer := 8;  -- Input is INT8
    constant OUTPUT_ELEMENT_BW  : integer := 18; -- Expected *logical* output bitwidth (INT18 accumulator) <<<--- VERIFY Accumulator Type!
    constant PE_IN              : integer := 1;  -- Input PE <<< --- VERIFY
    constant PE_OUT             : integer := 1;  -- Output PE <<< --- VERIFY

    -- === Stream Length Calculation ===
    constant NUM_INPUT_SPATIAL_ELEMENTS : integer := 665; -- N*H*W for sliced input (1*665*1)
    constant INPUT_CHANNELS             : integer := 1;   -- Sliced input has C=1
    constant INPUT_SIMD                 : integer := INPUT_CHANNELS / PE_IN; -- = 1
    constant NUM_INPUT_WORDS            : integer := NUM_INPUT_SPATIAL_ELEMENTS * INPUT_SIMD; -- = 665

    constant NUM_OUTPUT_SPATIAL_ELEMENTS: integer := 1;  -- N*H*W for output (1*1*1)
    constant OUTPUT_CHANNELS            : integer := 5;  -- Final output classes
    constant OUTPUT_SIMD                : integer := OUTPUT_CHANNELS / PE_OUT; -- = 5
    constant NUM_OUTPUT_WORDS           : integer := NUM_OUTPUT_SPATIAL_ELEMENTS * OUTPUT_SIMD; -- = 5

    -- Input/Output Files
    constant INPUT_DATA_FILE  : string := "tb_input_data.txt";
    constant EXPECTED_DATA_FILE : string := "tb_expected_output.txt";
    constant OUTPUT_DATA_FILE : string := "tb_hw_output.txt";

    -- Simulation Control
    constant DRAIN_CYCLES       : integer := 500; -- Extra cycles after I/O finished

    -- Signals
    signal clk              : std_logic := '0';
    signal rst_n            : std_logic := '0';
    signal sim_done         : boolean := false; -- Master simulation done flag
    signal errors           : integer := 0;     -- Shared error count
    signal stimulus_finished: boolean := false; -- Flag from input process
    signal monitor_finished : boolean := false; -- Flag from output process

    -- DUT Connections
    signal s_axis_tdata_dut : std_logic_vector(AXI_INPUT_WIDTH-1 downto 0) := (others => '0');
    signal s_axis_tvalid_dut: std_logic := '0';
    signal s_axis_tready_dut: std_logic; -- Output from DUT
    signal m_axis_tdata_dut : std_logic_vector(AXI_OUTPUT_WIDTH-1 downto 0); -- Output from DUT
    signal m_axis_tvalid_dut: std_logic; -- Output from DUT
    signal m_axis_tready_dut: std_logic := '0'; -- Input to DUT


    -- Component Declaration for the Device Under Test (DUT)
    component StreamingDataflowPartition_1_wrapper -- <<< --- Ensure this matches wrapper entity name
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

    -- Instantiate the DUT
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

    -- Clock generation
    clk_gen : process
    begin
        while not sim_done loop -- Generate clock until sim_done is true
            clk <= '0';
            wait for CLK_PERIOD / 2;
            clk <= '1';
            wait for CLK_PERIOD / 2;
        end loop;
        wait; -- Stop clock generation
    end process clk_gen;

    -- Reset generation
    reset_gen : process
    begin
        rst_n <= '0';
        wait for CLK_PERIOD * 5;
        rst_n <= '1';
        wait; -- Hold reset high
    end process reset_gen;

    -- Input stimulus process
    stimulus_proc : process
        file     infile     : text open read_mode is INPUT_DATA_FILE;
        variable inline     : line;
        variable file_value : integer;
        variable packed_word: std_logic_vector(AXI_INPUT_WIDTH-1 downto 0);
        variable i          : integer := 0;
    begin
        report "Stimulus Process Started. Expecting " & integer'image(NUM_INPUT_WORDS) & " words.";
        s_axis_tvalid_dut <= '0';
        s_axis_tdata_dut  <= (others => '0');
        wait until rst_n = '1'; -- Wait for reset to deassert
        wait for CLK_PERIOD;

        -- Check if file opened successfully

        while (not endfile(infile) and i < NUM_INPUT_WORDS) loop
             -- Check for simulation end signal
             if sim_done then exit; end if;

             if not endfile(infile) then
                readline(infile, inline);
                -- Handle potential empty lines gracefully
                if inline'length > 0 then
                    read(inline, file_value);
                    packed_word := std_logic_vector(to_signed(file_value, AXI_INPUT_WIDTH)); -- Assuming INT8 input

                    s_axis_tdata_dut <= packed_word;
                    s_axis_tvalid_dut <= '1';

                    -- Wait until DUT is ready OR simulation ends
                    wait until s_axis_tready_dut = '1' or sim_done = true;

                    -- Only increment if data was accepted before sim_done
                    if s_axis_tready_dut = '1' and not sim_done then
                        wait for CLK_PERIOD / 4; -- Hold data stable briefly after TREADY goes high (optional good practice)
                        wait until rising_edge(clk);
                        i := i + 1;
                    else
                        exit; -- Exit loop if sim_done asserted while waiting
                    end if;
                 else
                    report "Stimulus Warning: Skipped empty line in input file." severity warning;
                 end if;
            else
                report "Stimulus Error: End of input file reached prematurely!" severity error;
                exit; -- Exit loop
            end if;
        end loop;

        -- Deassert valid after loop (unless ended early by sim_done)
        if not sim_done then
           s_axis_tvalid_dut <= '0';
           s_axis_tdata_dut  <= (others => '0');
        end if;

        -- Check if all expected words were sent
        if i /= NUM_INPUT_WORDS then
            report "Stimulus Warning: Only sent " & integer'image(i) & " words, expected " & integer'image(NUM_INPUT_WORDS) severity warning;
        end if;

        report "Stimulus Process Finished.";
        stimulus_finished <= true; -- Signal completion
        wait; -- Keep process alive until simulation ends
    end process stimulus_proc;

    -- Output monitoring and checking process
    monitor_proc : process
        file     outfile     : text open write_mode is OUTPUT_DATA_FILE;
        file     expectedfile: text open read_mode is EXPECTED_DATA_FILE;
        variable outline     : line;
        variable expectedline: line;
        variable file_value  : integer;
        variable hw_value_signed : signed(AXI_OUTPUT_WIDTH-1 downto 0);
        variable expected_val_signed : signed(AXI_OUTPUT_WIDTH-1 downto 0);
        variable hw_value_int : integer;
        variable expected_val_int : integer;
        variable temp_unsigned : unsigned(AXI_OUTPUT_WIDTH-1 downto 0);
        variable i           : integer := 0;
        variable current_errors : integer := 0;

    begin
        report "Monitor Process Started. Expecting " & integer'image(NUM_OUTPUT_WORDS) & " words.";
        m_axis_tready_dut <= '0'; -- Start not ready
        wait until rst_n = '1';
        m_axis_tready_dut <= '1'; -- Be ready to receive

        -- Check if files opened

        monitor_loop : loop
            -- Exit loop if simulation is done or expected words received
            exit monitor_loop when sim_done = true or i >= NUM_OUTPUT_WORDS;

            -- Wait for valid data from DUT OR simulation end
            wait until m_axis_tvalid_dut = '1' or sim_done = true;

            -- Check if loop should exit due to sim_done
            exit monitor_loop when sim_done = true;

             -- Read data only if TVALID is high (and TREADY is high - which it is)
             if m_axis_tvalid_dut = '1' then
                 wait until rising_edge(clk); -- Sample on clock edge after TVALID detected
                 hw_value_signed := signed(m_axis_tdata_dut);

                 -- Convert HW signed to integer for writing/reporting
                 temp_unsigned := unsigned(hw_value_signed);
                 if hw_value_signed(AXI_OUTPUT_WIDTH-1) = '1' then
                     hw_value_int := to_integer(temp_unsigned) - 2**AXI_OUTPUT_WIDTH;
                 else
                     hw_value_int := to_integer(temp_unsigned);
                 end if;
                 write(outline, hw_value_int);
                 writeline(outfile, outline);

                 -- Read expected data
                 if not endfile(expectedfile) then
                      readline(expectedfile, expectedline);
                      if expectedline'length > 0 then
                         read(expectedline, file_value);
                         expected_val_signed := to_signed(file_value, AXI_OUTPUT_WIDTH);
                         -- Convert expected to integer for reporting
                         temp_unsigned := unsigned(expected_val_signed);
                         if expected_val_signed(AXI_OUTPUT_WIDTH-1) = '1' then
                             expected_val_int := to_integer(temp_unsigned) - 2**AXI_OUTPUT_WIDTH;
                         else
                             expected_val_int := to_integer(temp_unsigned);
                         end if;
                      else
                         report "Monitor Warning: Skipped empty line in expected file." severity warning;
                         expected_val_signed := (others => 'X'); expected_val_int := integer'high;
                      end if;
                 else
                      report "Monitor Error: Ran out of expected data!" severity error;
                      expected_val_signed := (others => 'X'); expected_val_int := integer'high;
                      current_errors := current_errors + 1; -- Count missing expected data as error
                      -- Optionally exit loop early if expected data runs out
                      -- exit monitor_loop;
                 end if;

                 -- Compare signed values directly
                 if hw_value_signed /= expected_val_signed then
                       report "Monitor Error: Mismatch at output word " & integer'image(i) &
                              ". Expected=" & integer'image(expected_val_int) &
                              ", Got=" & integer'image(hw_value_int) severity warning;
                       current_errors := current_errors + 1;
                 end if;

                 i := i + 1; -- Increment only after successfully reading and comparing
            else
                -- If TVALID was not high, just wait for next clock edge
                wait until rising_edge(clk);
            end if;

        end loop monitor_loop;

        m_axis_tready_dut <= '0'; -- Stop accepting data
        report "Monitor Process Finished. Received " & integer'image(i) & " words.";
        errors <= current_errors; -- Update shared signal
        monitor_finished <= true; -- Signal completion
        wait; -- Keep process alive until simulation ends
    end process monitor_proc;

    -- Main simulation control process
    sim_control_proc : process
        -- Timeout definition (adjust as needed)
        constant SIM_TIMEOUT_CYCLES : integer := 100000; -- Generous timeout
        variable current_cycle    : integer := 0;
    begin
        report "Simulation Control Process Started.";
        -- Wait until both stimulus and monitor signal completion OR timeout occurs
        loop
            wait for CLK_PERIOD;
            current_cycle := current_cycle + 1;
            -- Exit if both finished or timeout reached
            exit when stimulus_finished = true and monitor_finished = true;
            exit when current_cycle >= SIM_TIMEOUT_CYCLES;
        end loop;

        if current_cycle >= SIM_TIMEOUT_CYCLES then
             report "Simulation TIMEOUT after " & integer'image(current_cycle) & " cycles!" severity error;
        else
             report "Simulation completed normally based on I/O flags." severity note;
        end if;

        -- Add a few extra cycles for final outputs/reports
        wait for CLK_PERIOD * 10;

        report "Ending Simulation.";
        sim_done <= true; -- Signal simulation end to other processes
        wait; -- End this process
    end process sim_control_proc;


end behavior;
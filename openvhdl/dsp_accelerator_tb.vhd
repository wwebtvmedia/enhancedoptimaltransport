library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;
use std.textio.all;

entity dsp_accelerator_tb is
end dsp_accelerator_tb;

architecture Behavioral of dsp_accelerator_tb is

    -- Component Declaration
    component dsp_accelerator
        generic (
            C_S_AXI_DATA_WIDTH : integer := 32;
            C_S_AXI_ADDR_WIDTH : integer := 7;
            C_M_AXI_ADDR_WIDTH : integer := 32;
            C_M_AXI_DATA_WIDTH : integer := 32
        );
        port (
            clk           : in  std_logic;
            reset_n       : in  std_logic;
            s_axi_awaddr  : in  std_logic_vector(C_S_AXI_ADDR_WIDTH-1 downto 0);
            s_axi_awvalid : in  std_logic;
            s_axi_awready : out std_logic;
            s_axi_wdata   : in  std_logic_vector(C_S_AXI_DATA_WIDTH-1 downto 0);
            s_axi_wstrb   : in  std_logic_vector((C_S_AXI_DATA_WIDTH/8)-1 downto 0);
            s_axi_wvalid  : in  std_logic;
            s_axi_wready  : out std_logic;
            s_axi_bresp   : out std_logic_vector(1 downto 0);
            s_axi_bvalid  : out std_logic;
            s_axi_bready  : in  std_logic;
            s_axi_araddr  : in  std_logic_vector(C_S_AXI_ADDR_WIDTH-1 downto 0);
            s_axi_arvalid : in  std_logic;
            s_axi_arready : out std_logic;
            s_axi_rdata   : out std_logic_vector(C_S_AXI_DATA_WIDTH-1 downto 0);
            s_axi_rresp   : out std_logic_vector(1 downto 0);
            s_axi_rvalid  : out std_logic;
            s_axi_rready  : in  std_logic;
            m_axi_araddr  : out std_logic_vector(C_M_AXI_ADDR_WIDTH-1 downto 0);
            m_axi_arvalid : out std_logic;
            m_axi_arready : in  std_logic;
            m_axi_rdata   : in  std_logic_vector(C_M_AXI_DATA_WIDTH-1 downto 0);
            m_axi_rvalid  : in  std_logic;
            m_axi_rready  : out std_logic;
            m_axi_awaddr  : out std_logic_vector(C_M_AXI_ADDR_WIDTH-1 downto 0);
            m_axi_awvalid : out std_logic;
            m_axi_awready : in  std_logic;
            m_axi_wdata   : out std_logic_vector(C_M_AXI_DATA_WIDTH-1 downto 0);
            m_axi_wvalid  : out std_logic;
            m_axi_wready  : in  std_logic;
            m_axi_bvalid  : in  std_logic;
            m_axi_bready  : out std_logic;
            interrupt     : out std_logic
        );
    end component;

    -- Constants
    constant clk_period : time := 10 ns;

    -- Memory Model (Emulated RAM)
    -- Layout:
    -- 0x00000 - 0x03FFF: Input Image (A) - 16K words
    -- 0x04000 - 0x07FFF: Weights         - 16K words
    -- 0x08000 - 0x0BFFF: Output Buffer   - 16K words
    -- 0x0C000 - 0x0CFFF: Bias            - 4K words
    type mem_array is array (0 to 65535) of std_logic_vector(31 downto 0);
    signal system_memory : mem_array := (others => (others => '0'));

    -- Signals
    signal clk           : std_logic := '0';
    signal reset_n       : std_logic := '0';
    signal s_axi_awaddr  : std_logic_vector(6 downto 0) := (others => '0');
    signal s_axi_awvalid, s_axi_wvalid, s_axi_bready, s_axi_arvalid, s_axi_rready : std_logic := '0';
    signal s_axi_wdata : std_logic_vector(31 downto 0);
    signal m_axi_araddr, m_axi_awaddr : std_logic_vector(31 downto 0);
    signal m_axi_arvalid, m_axi_awvalid, m_axi_wvalid : std_logic;
    signal m_axi_arready, m_axi_awready, m_axi_wready, m_axi_rvalid, m_axi_bvalid : std_logic := '0';
    signal m_axi_rdata, m_axi_wdata, s_axi_rdata : std_logic_vector(31 downto 0);
    signal interrupt : std_logic;

begin

    uut: dsp_accelerator port map (
        clk => clk, reset_n => reset_n,
        s_axi_awaddr => s_axi_awaddr, s_axi_awvalid => s_axi_awvalid, s_axi_awready => open,
        s_axi_wdata => s_axi_wdata, s_axi_wstrb => "1111", s_axi_wvalid => s_axi_wvalid, s_axi_wready => open,
        s_axi_bresp => open, s_axi_bvalid => open, s_axi_bready => s_axi_bready,
        s_axi_araddr => (others => '0'), s_axi_arvalid => s_axi_arvalid, s_axi_arready => open,
        s_axi_rdata => s_axi_rdata, s_axi_rresp => open, s_axi_rvalid => open, s_axi_rready => s_axi_rready,
        m_axi_araddr => m_axi_araddr, m_axi_arvalid => m_axi_arvalid, m_axi_arready => m_axi_arready,
        m_axi_rdata => m_axi_rdata, m_axi_rvalid => m_axi_rvalid, m_axi_rready => open,
        m_axi_awaddr => m_axi_awaddr, m_axi_awvalid => m_axi_awvalid, m_axi_awready => m_axi_awready,
        m_axi_wdata => m_axi_wdata, m_axi_wvalid => m_axi_wvalid, m_axi_wready => m_axi_wready,
        m_axi_bvalid => m_axi_bvalid, m_axi_bready => open, interrupt => interrupt
    );

    clk_process : process begin
        clk <= '0'; wait for clk_period/2;
        clk <= '1'; wait for clk_period/2;
    end process;

    dma_slave_proc: process(clk)
        variable addr_idx : integer;
    begin
        if rising_edge(clk) then
            if m_axi_arvalid = '1' then
                m_axi_arready <= '1';
                addr_idx := to_integer(unsigned(m_axi_araddr)) / 4;
                m_axi_rdata <= system_memory(addr_idx mod 65536);
                m_axi_rvalid <= '1';
            else
                m_axi_arready <= '0'; m_axi_rvalid <= '0';
            end if;
            if m_axi_awvalid = '1' and m_axi_wvalid = '1' then
                m_axi_awready <= '1'; m_axi_wready <= '1';
                addr_idx := to_integer(unsigned(m_axi_awaddr)) / 4;
                system_memory(addr_idx mod 65536) <= m_axi_wdata;
                m_axi_bvalid <= '1';
            else
                m_axi_awready <= '0'; m_axi_wready <= '0'; m_axi_bvalid <= '0';
            end if;
        end if;
    end process;

    stim_proc: process
        procedure axi_write(addr : std_logic_vector(6 downto 0); data : std_logic_vector(31 downto 0)) is
        begin
            s_axi_awaddr <= addr; s_axi_wdata <= data;
            s_axi_awvalid <= '1'; s_axi_wvalid <= '1'; s_axi_bready <= '1';
            wait until rising_edge(clk) and m_axi_bvalid = '0'; -- dummy wait
            wait for clk_period;
            s_axi_awvalid <= '0'; s_axi_wvalid <= '0';
        end procedure;
    begin
        -- 1. Initialize RAM with neural parameters
        -- Input Image: 12x12 gradient
        for i in 0 to 143 loop
            system_memory(i) <= std_logic_vector(to_unsigned(i, 32));
        end loop;
        -- Weights: Identity (center pixel = 1)
        for i in 0 to 8 loop
            if i = 4 then system_memory(16384 + i) <= x"00000001";
            else system_memory(16384 + i) <= x"00000000"; end if;
        end loop;

        reset_n <= '0'; wait for 100 ns; reset_n <= '1'; wait for clk_period*5;

        -- 2. Configure Conv2D Operation
        axi_write("0001000", x"00000008"); -- Opcode 8: Conv2D
        axi_write("1000100", x"00000001"); -- In Channels: 1
        axi_write("1001000", x"00000001"); -- Out Channels: 1
        axi_write("1001100", x"0000000C"); -- In Height: 12
        axi_write("1010000", x"0000000C"); -- In Width: 12
        axi_write("1010100", x"00000003"); -- K Height: 3
        axi_write("1011000", x"00000003"); -- K Width: 3
        
        axi_write("0011000", x"00000000"); -- Addr A (Input): 0
        axi_write("1100000", x"00010000"); -- Addr W (Weights): 16384*4
        axi_write("0100000", x"00020000"); -- Addr Out: 32768*4
        
        -- Start Execution
        axi_write("0000000", x"00000001");

        wait until interrupt = '1';
        report "Conv2D Completed. Checking result at RAM[32768]...";
        assert system_memory(32768) /= x"00000000" report "Error: Conv2D output is zero!" severity error;
        
        -- 3. Test Opcode 9: Conditioning (Broadcast Add)
        report "Starting Opcode 9: Conditioning Test...";
        -- Input Image at RAM[0], Vector at RAM[16384], Output at RAM[49152]
        system_memory(16384) <= x"0000000A"; -- Bias of 10 for channel 0
        
        axi_write("0001000", x"00000009"); -- Opcode 9: Cond
        axi_write("0011000", x"00000000"); -- Addr A (Feature Map): 0
        axi_write("0011100", x"00010000"); -- Addr B (Vector): 16384*4
        axi_write("0100000", x"00030000"); -- Addr Out: 49152*4
        
        axi_write("0000000", x"00000001"); -- Start
        wait until interrupt = '1';
        
        report "Conditioning Completed. Checking result at RAM[49152]...";
        -- First pixel should be 0 + 10 = 10 (x"A")
        assert system_memory(49152) = x"0000000A" report "Error: Conditioning output incorrect! Expected 10, got " & integer'image(to_integer(unsigned(system_memory(49152)))) severity error;

        report "Simulation Success: VHDL aligned with OpenCL Memory-Mapped Parameters.";
        wait;
    end process;

end Behavioral;

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
            C_S_AXI_ADDR_WIDTH : integer := 6;
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
    constant IMG_WIDTH  : integer := 128;
    constant IMG_HEIGHT : integer := 128;
    constant IMG_SIZE   : integer := IMG_WIDTH * IMG_HEIGHT;
    constant clk_period : time := 10 ns;

    -- Memory Model (Grayscale Image)
    type mem_array is array (0 to 32767) of std_logic_vector(31 downto 0);
    signal system_memory : mem_array := (others => (others => '0'));

    -- Signals
    signal clk           : std_logic := '0';
    signal reset_n       : std_logic := '0';
    
    -- AXI Lite
    signal s_axi_awaddr  : std_logic_vector(5 downto 0) := (others => '0');
    signal s_axi_awvalid : std_logic := '0';
    signal s_axi_awready : std_logic;
    signal s_axi_wdata   : std_logic_vector(31 downto 0) := (others => '0');
    signal s_axi_wvalid  : std_logic := '0';
    signal s_axi_wready  : std_logic;
    signal s_axi_bready  : std_logic := '0';
    signal s_axi_bvalid  : std_logic;
    signal s_axi_araddr  : std_logic_vector(5 downto 0) := (others => '0');
    signal s_axi_arvalid : std_logic := '0';
    signal s_axi_arready : std_logic;
    signal s_axi_rready  : std_logic := '0';
    signal s_axi_rvalid  : std_logic;
    signal s_axi_rdata   : std_logic_vector(31 downto 0);

    -- AXI Master (DMA)
    signal m_axi_araddr  : std_logic_vector(31 downto 0);
    signal m_axi_arvalid : std_logic;
    signal m_axi_arready : std_logic := '0';
    signal m_axi_rdata   : std_logic_vector(31 downto 0) := (others => '0');
    signal m_axi_rvalid  : std_logic := '0';
    signal m_axi_rready  : std_logic;
    
    signal m_axi_awaddr  : std_logic_vector(31 downto 0);
    signal m_axi_awvalid : std_logic;
    signal m_axi_awready : std_logic := '0';
    signal m_axi_wdata   : std_logic_vector(31 downto 0);
    signal m_axi_wvalid  : std_logic;
    signal m_axi_wready  : std_logic := '0';
    signal m_axi_bvalid  : std_logic := '0';
    signal m_axi_bready  : std_logic;
    
    signal interrupt     : std_logic;

begin

    -- Instantiate the Unit Under Test (UUT)
    uut: dsp_accelerator
        port map (
            clk           => clk,
            reset_n       => reset_n,
            s_axi_awaddr  => s_axi_awaddr,
            s_axi_awvalid => s_axi_awvalid,
            s_axi_awready => s_axi_awready,
            s_axi_wdata   => s_axi_wdata,
            s_axi_wstrb   => "1111",
            s_axi_wvalid  => s_axi_wvalid,
            s_axi_wready  => s_axi_wready,
            s_axi_bresp   => open,
            s_axi_bvalid  => s_axi_bvalid,
            s_axi_bready  => s_axi_bready,
            s_axi_araddr  => s_axi_araddr,
            s_axi_arvalid => s_axi_arvalid,
            s_axi_arready => s_axi_arready,
            s_axi_rdata   => s_axi_rdata,
            s_axi_rresp   => open,
            s_axi_rvalid  => s_axi_rvalid,
            s_axi_rready  => s_axi_rready,
            m_axi_araddr  => m_axi_araddr,
            m_axi_arvalid => m_axi_arvalid,
            m_axi_arready => m_axi_arready,
            m_axi_rdata   => m_axi_rdata,
            m_axi_rvalid  => m_axi_rvalid,
            m_axi_rready  => m_axi_rready,
            m_axi_awaddr  => m_axi_awaddr,
            m_axi_awvalid => m_axi_awvalid,
            m_axi_awready => m_axi_awready,
            m_axi_wdata   => m_axi_wdata,
            m_axi_wvalid  => m_axi_wvalid,
            m_axi_wready  => m_axi_wready,
            m_axi_bvalid  => m_axi_bvalid,
            m_axi_bready  => m_axi_bready,
            interrupt     => interrupt
        );

    -- Clock process
    clk_process :process
    begin
        clk <= '0';
        wait for clk_period/2;
        clk <= '1';
        wait for clk_period/2;
    end process;

    -- DMA Slave Emulation with Memory Array
    dma_slave_proc: process(clk)
        variable addr_idx : integer;
    begin
        if rising_edge(clk) then
            -- Handle Read Requests
            if m_axi_arvalid = '1' then
                m_axi_arready <= '1';
                addr_idx := to_integer(unsigned(m_axi_araddr)) / 4;
                if addr_idx < 32768 then
                    m_axi_rdata <= system_memory(addr_idx);
                else
                    m_axi_rdata <= (others => '0');
                end if;
                m_axi_rvalid <= '1';
            else
                m_axi_arready <= '0';
                m_axi_rvalid <= '0';
            end if;
            
            -- Handle Write Requests
            if m_axi_awvalid = '1' and m_axi_wvalid = '1' then
                m_axi_awready <= '1';
                m_axi_wready <= '1';
                addr_idx := to_integer(unsigned(m_axi_awaddr)) / 4;
                if addr_idx < 32768 then
                    system_memory(addr_idx) <= m_axi_wdata;
                end if;
                m_axi_bvalid <= '1';
            else
                m_axi_awready <= '0';
                m_axi_wready <= '0';
                m_axi_bvalid <= '0';
            end if;
        end if;
    end process;

    -- Stimulus process
    stim_proc: process
        file f_out : text open write_mode is "output_image.txt";
        variable l_out : line;
        variable val : integer;
    begin		
        -- Initialize Input A Memory with a Gradient (0 to 127)
        for i in 0 to IMG_SIZE-1 loop
            val := i mod 256;
            system_memory(i) <= std_logic_vector(to_unsigned(val, 32));
        end loop;
        
        -- Initialize Input B Memory with Constant 2 (for multiplying)
        for i in 0 to IMG_SIZE-1 loop
            system_memory(16384 + i) <= std_logic_vector(to_unsigned(2, 32));
        end loop;

        -- Reset
        reset_n <= '0';
        wait for 100 ns;	
        reset_n <= '1';
        wait for clk_period*10;

        -- 1. Configure for Image Processing (Multiply Gradient by 2)
        
        -- Set Opcode: 2 (Mul)
        s_axi_awaddr <= "001000"; -- 0x08
        s_axi_wdata <= x"00000002";
        s_axi_awvalid <= '1'; s_axi_wvalid <= '1'; s_axi_bready <= '1';
        wait until s_axi_bvalid = '1';
        s_axi_awvalid <= '0'; s_axi_wvalid <= '0';
        wait for clk_period;

        -- Set Vector Length: IMG_SIZE
        s_axi_awaddr <= "001100"; -- 0x0C
        s_axi_wdata <= std_logic_vector(to_unsigned(IMG_SIZE, 32));
        s_axi_awvalid <= '1'; s_axi_wvalid <= '1';
        wait until s_axi_bvalid = '1';
        s_axi_awvalid <= '0'; s_axi_wvalid <= '0';
        wait for clk_period;
        
        -- Set Input A Addr: 0
        s_axi_awaddr <= "011000"; -- 0x18
        s_axi_wdata <= x"00000000";
        s_axi_awvalid <= '1'; s_axi_wvalid <= '1';
        wait until s_axi_bvalid = '1';
        s_axi_awvalid <= '0'; s_axi_wvalid <= '0';
        wait for clk_period;
        
        -- Set Input B Addr: 16384*4 = 0x10000
        s_axi_awaddr <= "011100"; -- 0x1C
        s_axi_wdata <= x"00010000";
        s_axi_awvalid <= '1'; s_axi_wvalid <= '1';
        wait until s_axi_bvalid = '1';
        s_axi_awvalid <= '0'; s_axi_wvalid <= '0';
        wait for clk_period;
        
        -- Set Output Addr: 8192*4 = 0x08000
        s_axi_awaddr <= "100000"; -- 0x20
        s_axi_wdata <= x"00008000";
        s_axi_awvalid <= '1'; s_axi_wvalid <= '1';
        wait until s_axi_bvalid = '1';
        s_axi_awvalid <= '0'; s_axi_wvalid <= '0';
        wait for clk_period;

        -- Set Start Bit
        s_axi_awaddr <= "000000"; -- 0x00
        s_axi_wdata <= x"00000001";
        s_axi_awvalid <= '1'; s_axi_wvalid <= '1';
        wait until s_axi_bvalid = '1';
        s_axi_awvalid <= '0'; s_axi_wvalid <= '0';

        -- 2. Wait for completion
        wait until interrupt = '1';
        
        -- 3. Dump Output Memory to File
        report "Dumping output memory to output_image.txt...";
        for i in 0 to IMG_SIZE-1 loop
            write(l_out, to_integer(unsigned(system_memory(8192 + i))));
            writeline(f_out, l_out);
        end loop;

        report "Simulation Completed. Image generated in memory and dumped.";
        wait;
    end process;

end Behavioral;

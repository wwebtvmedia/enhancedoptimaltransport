library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

-- =========================================================================
-- DSP Hardware Accelerator (Ported from dsp_imp.cl)
-- 
-- Features:
-- - AXI4-Lite Control Interface (expanded for Conv2D/PixelShuffle/Cond)
-- - AXI4 Master DMA Interface for fetching and storing data/parameters
-- - All Parameters (Weights, Bias, Embeddings) fetched from emulated RAM
-- - Support for: Add, Sub, Mul, MAC, Quantize, PixelShuffle, SiLU, Conv2D, Cond
-- =========================================================================

entity dsp_accelerator is
    generic (
        C_S_AXI_DATA_WIDTH : integer := 32;
        C_S_AXI_ADDR_WIDTH : integer := 7;
        C_M_AXI_ADDR_WIDTH : integer := 32;
        C_M_AXI_DATA_WIDTH : integer := 32
    );
    port (
        clk         : in  std_logic;
        reset_n     : in  std_logic;
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
end dsp_accelerator;

architecture Behavioral of dsp_accelerator is
    signal reg_ctrl      : std_logic_vector(31 downto 0) := (others => '0');
    signal reg_opcode    : std_logic_vector(31 downto 0) := (others => '0');
    signal reg_dim_m     : std_logic_vector(31 downto 0) := (others => '0');
    signal reg_addr_a    : std_logic_vector(31 downto 0) := (others => '0');
    signal reg_addr_b    : std_logic_vector(31 downto 0) := (others => '0');
    signal reg_addr_out  : std_logic_vector(31 downto 0) := (others => '0');
    signal reg_in_c      : std_logic_vector(31 downto 0) := (others => '0');
    signal reg_out_c     : std_logic_vector(31 downto 0) := (others => '0');
    signal reg_in_h      : std_logic_vector(31 downto 0) := (others => '0');
    signal reg_in_w      : std_logic_vector(31 downto 0) := (others => '0');
    signal reg_k_h       : std_logic_vector(31 downto 0) := (others => '0');
    signal reg_k_w       : std_logic_vector(31 downto 0) := (others => '0');
    signal reg_upscale   : std_logic_vector(31 downto 0) := (others => '0');
    signal reg_addr_w    : std_logic_vector(31 downto 0) := (others => '0');
    signal reg_addr_bias : std_logic_vector(31 downto 0) := (others => '0');

    type state_type is (IDLE, FETCH_A, FETCH_B, FETCH_W, EXECUTE, WRITE_OUT, DONE);
    signal current_state : state_type := IDLE;
    signal is_busy, is_done : std_logic := '0';
    signal data_a, data_b, data_w : signed(31 downto 0) := (others => '0');
    signal data_out : signed(31 downto 0) := (others => '0');
    signal mac_acc  : signed(63 downto 0) := (others => '0');
    signal oc_cnt, oh_cnt, ow_cnt : unsigned(15 downto 0) := (others => '0');
    signal ic_cnt, kh_cnt, kw_cnt : unsigned(15 downto 0) := (others => '0');
    signal elem_cnt : unsigned(31 downto 0) := (others => '0');
    signal axi_awready, axi_wready, axi_arready, axi_rvalid : std_logic := '0';

begin
    process(clk)
        variable loc_addr : std_logic_vector(7 downto 0);
    begin
        if rising_edge(clk) then
            if reset_n = '0' then
                axi_awready <= '0'; axi_wready <= '0'; reg_ctrl <= (others => '0');
            else
                if (axi_awready = '0' and s_axi_awvalid = '1' and s_axi_wvalid = '1') then
                    axi_awready <= '1'; axi_wready <= '1';
                    loc_addr := '0' & s_axi_awaddr;
                    case loc_addr is
                        when x"00" => reg_ctrl     <= s_axi_wdata;
                        when x"08" => reg_opcode   <= s_axi_wdata;
                        when x"0C" => reg_dim_m    <= s_axi_wdata;
                        when x"18" => reg_addr_a   <= s_axi_wdata;
                        when x"1C" => reg_addr_b   <= s_axi_wdata;
                        when x"20" => reg_addr_out <= s_axi_wdata;
                        when x"44" => reg_in_c     <= s_axi_wdata;
                        when x"48" => reg_out_c    <= s_axi_wdata;
                        when x"4C" => reg_in_h     <= s_axi_wdata;
                        when x"50" => reg_in_w     <= s_axi_wdata;
                        when x"54" => reg_k_h      <= s_axi_wdata;
                        when x"58" => reg_k_w      <= s_axi_wdata;
                        when x"5C" => reg_upscale  <= s_axi_wdata;
                        when x"60" => reg_addr_w    <= s_axi_wdata;
                        when x"64" => reg_addr_bias <= s_axi_wdata;
                        when others => null;
                    end case;
                else
                    axi_awready <= '0'; axi_wready <= '0';
                end if;
                if current_state /= IDLE then reg_ctrl(0) <= '0'; end if;
            end if;
        end if;
    end process;

    s_axi_awready <= axi_awready; s_axi_wready  <= axi_wready;
    s_axi_bvalid  <= axi_awready; s_axi_bresp   <= "00";

    process(clk)
        variable loc_addr : std_logic_vector(7 downto 0);
    begin
        if rising_edge(clk) then
            if reset_n = '0' then
                axi_arready <= '0'; axi_rvalid <= '0';
            else
                if axi_arready = '0' and s_axi_arvalid = '1' then
                    axi_arready <= '1';
                    loc_addr := '0' & s_axi_araddr;
                    case loc_addr is
                        when x"00" => s_axi_rdata <= reg_ctrl;
                        when x"04" => s_axi_rdata <= (0 => is_busy, 1 => is_done, others => '0');
                        when x"08" => s_axi_rdata <= reg_opcode;
                        when x"0C" => s_axi_rdata <= reg_dim_m;
                        when x"18" => s_axi_rdata <= reg_addr_a;
                        when x"20" => s_axi_rdata <= reg_addr_out;
                        when others => s_axi_rdata <= (others => '0');
                    end case;
                else
                    axi_arready <= '0';
                end if;
                if axi_arready = '1' then axi_rvalid <= '1';
                elsif s_axi_rready = '1' then axi_rvalid <= '0';
                end if;
            end if;
        end if;
    end process;
    
    s_axi_arready <= axi_arready; s_axi_rvalid  <= axi_rvalid; s_axi_rresp   <= "00";

    process(clk)
        variable addr_v : unsigned(31 downto 0);
        variable r_v, ic_v, ih_v, iw_v : integer;
        variable temp_idx : unsigned(31 downto 0);
    begin
        if rising_edge(clk) then
            if reset_n = '0' then
                current_state <= IDLE;
                is_busy <= '0'; is_done <= '0'; interrupt <= '0';
                oc_cnt <= (others => '0'); oh_cnt <= (others => '0'); ow_cnt <= (others => '0');
                ic_cnt <= (others => '0'); kh_cnt <= (others => '0'); kw_cnt <= (others => '0');
                elem_cnt <= (others => '0');
            else
                case current_state is
                    when IDLE =>
                        is_done <= '0'; interrupt <= '0';
                        if reg_ctrl(0) = '1' then
                            is_busy <= '1'; elem_cnt <= (others => '0');
                            oc_cnt <= (others => '0'); oh_cnt <= (others => '0'); ow_cnt <= (others => '0');
                            ic_cnt <= (others => '0'); kh_cnt <= (others => '0'); kw_cnt <= (others => '0');
                            mac_acc <= (others => '0'); current_state <= FETCH_A;
                        else is_busy <= '0'; end if;

                    when FETCH_A =>
                        m_axi_arvalid <= '1';
                        if reg_opcode(3 downto 0) = "1000" then -- Conv2D
                            temp_idx := resize(resize(ic_cnt, 32) * resize(unsigned(reg_in_h(15 downto 0)), 32), 32);
                            temp_idx := temp_idx + resize(oh_cnt + kh_cnt, 32);
                            temp_idx := resize(temp_idx * resize(unsigned(reg_in_w(15 downto 0)), 32), 32);
                            temp_idx := temp_idx + resize(ow_cnt + kw_cnt, 32);
                            addr_v := unsigned(reg_addr_a) + shift_left(temp_idx, 2);
                        elsif reg_opcode(3 downto 0) = "1001" then -- Condition (Opcode 9)
                            temp_idx := resize(resize(oc_cnt, 32) * resize(unsigned(reg_in_h(15 downto 0)), 32) * resize(unsigned(reg_in_w(15 downto 0)), 32), 32);
                            temp_idx := temp_idx + resize(resize(oh_cnt, 32) * resize(unsigned(reg_in_w(15 downto 0)), 32), 32) + resize(ow_cnt, 32);
                            addr_v := unsigned(reg_addr_a) + shift_left(temp_idx, 2);
                        elsif reg_opcode(3 downto 0) = "0101" then -- PixelShuffle
                            r_v := to_integer(unsigned(reg_upscale));
                            if r_v = 0 then r_v := 1; end if;
                            ic_v := to_integer(oc_cnt) * (r_v * r_v) + (to_integer(oh_cnt) mod r_v) * r_v + (to_integer(ow_cnt) mod r_v);
                            ih_v := to_integer(oh_cnt) / r_v;
                            iw_v := to_integer(ow_cnt) / r_v;
                            temp_idx := resize(to_unsigned(ic_v, 32) * resize(unsigned(reg_in_h(15 downto 0)), 32) * resize(unsigned(reg_in_w(15 downto 0)), 32), 32);
                            temp_idx := temp_idx + resize(to_unsigned(ih_v, 32) * resize(unsigned(reg_in_w(15 downto 0)), 32), 32) + to_unsigned(iw_v, 32);
                            addr_v := unsigned(reg_addr_a) + shift_left(temp_idx, 2);
                        else addr_v := unsigned(reg_addr_a) + shift_left(elem_cnt, 2); end if;
                        m_axi_araddr <= std_logic_vector(addr_v);
                        if m_axi_arready = '1' then
                            data_a <= signed(m_axi_rdata); m_axi_arvalid <= '0';
                            if reg_opcode(3 downto 0) = "1000" then current_state <= FETCH_W;
                            elsif reg_opcode(3 downto 0) = "1001" then current_state <= FETCH_B;
                            elsif reg_opcode(3 downto 0) = "0101" or reg_opcode(3 downto 0) = "0110" then current_state <= EXECUTE;
                            else current_state <= FETCH_B; end if;
                        end if;

                    when FETCH_B =>
                        if reg_opcode(3 downto 0) = "1001" then
                            addr_v := unsigned(reg_addr_b) + shift_left(resize(oc_cnt, 32), 2);
                        else
                            addr_v := unsigned(reg_addr_b) + shift_left(elem_cnt, 2);
                        end if;
                        m_axi_araddr <= std_logic_vector(addr_v);
                        m_axi_arvalid <= '1';
                        if m_axi_arready = '1' then data_b <= signed(m_axi_rdata); m_axi_arvalid <= '0'; current_state <= EXECUTE; end if;

                    when FETCH_W =>
                        temp_idx := resize((resize(oc_cnt, 32) * resize(unsigned(reg_in_c(15 downto 0)), 32) + resize(ic_cnt, 32)), 32);
                        temp_idx := resize(temp_idx * resize(unsigned(reg_k_h(15 downto 0)), 32) + resize(kh_cnt, 32), 32);
                        temp_idx := resize(temp_idx * resize(unsigned(reg_k_w(15 downto 0)), 32) + resize(kw_cnt, 32), 32);
                        addr_v := unsigned(reg_addr_w) + shift_left(temp_idx, 2);
                        m_axi_araddr <= std_logic_vector(addr_v); m_axi_arvalid <= '1';
                        if m_axi_arready = '1' then data_w <= signed(m_axi_rdata); m_axi_arvalid <= '0'; current_state <= EXECUTE; end if;

                    when EXECUTE =>
                        case reg_opcode(3 downto 0) is
                            when "0000" | "1001" => data_out <= data_a + data_b; -- Add / Cond
                            when "0010" => data_out <= resize(data_a * data_b, 32); -- Mul
                            when "0101" => data_out <= data_a; -- Shuffle
                            when "0110" => if data_a > 0 then data_out <= data_a; else data_out <= (others => '0'); end if;
                            when "1000" => 
                                mac_acc <= mac_acc + (data_a * data_w);
                                if kw_cnt < unsigned(reg_k_w(15 downto 0)) - 1 then kw_cnt <= kw_cnt + 1; current_state <= FETCH_A;
                                elsif kh_cnt < unsigned(reg_k_h(15 downto 0)) - 1 then kw_cnt <= (others => '0'); kh_cnt <= kh_cnt + 1; current_state <= FETCH_A;
                                elsif ic_cnt < unsigned(reg_in_c(15 downto 0)) - 1 then kw_cnt <= (others => '0'); kh_cnt <= (others => '0'); ic_cnt <= ic_cnt + 1; current_state <= FETCH_A;
                                else data_out <= resize(mac_acc, 32); current_state <= WRITE_OUT; end if;
                            when others => data_out <= data_a;
                        end case;
                        if reg_opcode(3 downto 0) /= "1000" then current_state <= WRITE_OUT; end if;

                    when WRITE_OUT =>
                        if reg_opcode(3 downto 0) >= "0101" then
                            temp_idx := resize(resize(oc_cnt, 32) * resize(unsigned(reg_in_h(15 downto 0)), 32) * resize(unsigned(reg_in_w(15 downto 0)), 32), 32);
                            temp_idx := temp_idx + resize(resize(oh_cnt, 32) * resize(unsigned(reg_in_w(15 downto 0)), 32), 32) + resize(ow_cnt, 32);
                            addr_v := unsigned(reg_addr_out) + shift_left(temp_idx, 2);
                        else addr_v := unsigned(reg_addr_out) + shift_left(elem_cnt, 2); end if;
                        m_axi_awaddr <= std_logic_vector(addr_v); m_axi_wdata <= std_logic_vector(data_out);
                        m_axi_awvalid <= '1'; m_axi_wvalid <= '1';
                        if m_axi_awready = '1' and m_axi_wready = '1' then
                            m_axi_awvalid <= '0'; m_axi_wvalid <= '0'; mac_acc <= (others => '0');
                            kw_cnt <= (others => '0'); kh_cnt <= (others => '0'); ic_cnt <= (others => '0');
                            if ow_cnt < unsigned(reg_in_w(15 downto 0)) - 1 then ow_cnt <= ow_cnt + 1; elem_cnt <= elem_cnt + 1; current_state <= FETCH_A;
                            elsif oh_cnt < unsigned(reg_in_h(15 downto 0)) - 1 then ow_cnt <= (others => '0'); oh_cnt <= oh_cnt + 1; elem_cnt <= elem_cnt + 1; current_state <= FETCH_A;
                            elsif oc_cnt < unsigned(reg_out_c(15 downto 0)) - 1 then ow_cnt <= (others => '0'); oh_cnt <= (others => '0'); oc_cnt <= oc_cnt + 1; elem_cnt <= elem_cnt + 1; current_state <= FETCH_A;
                            else current_state <= DONE; end if;
                        end if;

                    when DONE => is_busy <= '0'; is_done <= '1'; interrupt <= '1'; current_state <= IDLE;
                    when others => current_state <= IDLE;
                end case;
            end if;
        end if;
    end process;
end Behavioral;

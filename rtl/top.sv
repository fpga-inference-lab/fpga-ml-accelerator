// Basys 3 top: run one MLP inference per 20-byte request over USB-UART.
//
// Protocol (115200 baud, 8N1):
//   host -> FPGA: 20 bytes, the int8 input vector x[0..19]
//   FPGA -> host: 14 bytes
//     [0]      0xA5 (marks the start of a reply)
//     [1]      prediction (0 = down, 1 = up)
//     [2..5]   z2[0], int32 little-endian (down score)
//     [6..9]   z2[1], int32 little-endian (up score)
//     [10..13] compute cycles from start to done, uint32 little-endian
// If the host stops mid-request for TIMEOUT_CLKS cycles, the partial request is
// dropped so the next byte starts a fresh one.
//
// LEDs: 0 = last prediction, 1 = at least one result computed, 2 = busy,
//       8..4 = bytes received so far, 15 = heartbeat (bitstream is running).
module top #(
    parameter int CLKS_PER_BIT = 868,          // 100 MHz / 115200 baud
    parameter int TIMEOUT_CLKS = 10_000_000,   // 100 ms
    parameter HEX_DIR = "model/weights/hex/"
)(
    input  logic        clk,
    input  logic        btnC,       // reset
    input  logic        RsRx,       // from the host
    output logic        RsTx,       // to the host
    output logic [15:0] led
);

localparam int K1 = 20, O = 2, RESP_LEN = 14;

// the button is asynchronous to clk
logic rst_meta, rst;
always_ff @(posedge clk) begin
    rst_meta <= btnC;
    rst      <= rst_meta;
end

logic [7:0] rx_data;
logic       rx_valid;
logic [7:0] tx_data;
logic       tx_start, tx_busy;

uart_rx #(.CLKS_PER_BIT(CLKS_PER_BIT)) rx (
    .clk(clk), .rst(rst), .rx(RsRx), .data(rx_data), .valid(rx_valid)
);

uart_tx #(.CLKS_PER_BIT(CLKS_PER_BIT)) tx (
    .clk(clk), .rst(rst), .start(tx_start), .data(tx_data), .tx(RsTx), .busy(tx_busy)
);

logic signed [7:0]  x [K1];
logic signed [31:0] z2 [O];
logic               pred [1];
logic               mlp_start, mlp_done, mlp_done_d;

mlp #(.M(1), .K1(K1), .H(16), .O(O),
      .W1_FILE({HEX_DIR, "W1.hex"}), .B1_FILE({HEX_DIR, "b1.hex"}),
      .W2_FILE({HEX_DIR, "W2.hex"}), .B2_FILE({HEX_DIR, "b2.hex"})) net (
    .clk(clk), .rst(rst), .start(mlp_start), .x(x), .z2(z2), .pred(pred), .done(mlp_done)
);

typedef enum logic [1:0] {RECV, RUN, SEND} state_t;
state_t state;

logic [4:0]  rx_count;
logic [31:0] idle_clks;
logic [31:0] cycles;
logic [7:0]  resp [RESP_LEN];
logic [3:0]  resp_idx;
logic        have_result, last_pred;

always_ff @(posedge clk) mlp_done_d <= rst ? 1'b0 : mlp_done;

always_ff @(posedge clk) begin
    if (rst) begin
        state <= RECV;
        rx_count  <= 0;
        idle_clks <= 0;
        mlp_start <= 0;
        tx_start  <= 0;
        have_result <= 0;
        last_pred   <= 0;
    end else begin
        mlp_start <= 0;
        case (state)
            RECV: begin
                if (rx_valid) begin
                    x[rx_count] <= rx_data;
                    idle_clks <= 0;
                    if (rx_count == K1 - 1) begin
                        rx_count  <= 0;
                        mlp_start <= 1;
                        cycles    <= 0;
                        state     <= RUN;
                    end else
                        rx_count <= rx_count + 1;
                end else if (rx_count != 0) begin
                    idle_clks <= idle_clks + 1;
                    if (idle_clks == TIMEOUT_CLKS) begin
                        rx_count  <= 0;
                        idle_clks <= 0;
                    end
                end
            end

            // cycles counts every clock from the start pulse to done going high
            RUN: begin
                cycles <= cycles + 1;
                if (mlp_done && !mlp_done_d) begin
                    resp[0] <= 8'hA5;
                    resp[1] <= {7'b0, pred[0]};
                    for (int b = 0; b < 4; b++) begin
                        resp[2 + b]  <= z2[0][8*b +: 8];
                        resp[6 + b]  <= z2[1][8*b +: 8];
                        resp[10 + b] <= cycles[8*b +: 8];
                    end
                    last_pred   <= pred[0];
                    have_result <= 1;
                    resp_idx    <= 0;
                    state     <= SEND;
                end
            end

            SEND: begin
                if (tx_start)
                    tx_start <= 0;
                else if (!tx_busy) begin
                    if (resp_idx == RESP_LEN)
                        state <= RECV;
                    else begin
                        tx_data  <= resp[resp_idx];
                        tx_start <= 1;
                        resp_idx <= resp_idx + 1;
                    end
                end
            end
        endcase
    end
end

logic [26:0] heartbeat = 0;
always_ff @(posedge clk) heartbeat <= heartbeat + 1;

assign led[0]    = last_pred;
assign led[1]    = have_result;
assign led[2]    = state != RECV;
assign led[3]    = 1'b0;
assign led[8:4]  = rx_count;
assign led[14:9] = '0;
assign led[15]   = heartbeat[26];

endmodule

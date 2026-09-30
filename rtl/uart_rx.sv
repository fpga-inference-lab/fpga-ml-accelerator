// UART receiver, 8 data bits, no parity, 1 stop bit (8N1), LSB first.
// CLKS_PER_BIT = clock frequency / baud rate (100 MHz / 115200 = 868).
// Each bit is sampled once, in the middle of its bit period.
// valid pulses for one cycle when a byte arrives with a good stop bit.
module uart_rx #(
    parameter int CLKS_PER_BIT = 868
)(
    input  logic       clk,
    input  logic       rst,
    input  logic       rx,
    output logic [7:0] data,
    output logic       valid
);

// rx comes from outside the FPGA's clock domain: two flip-flops before use
logic rx_meta, rx_sync;
always_ff @(posedge clk) begin
    rx_meta <= rst ? 1'b1 : rx;
    rx_sync <= rst ? 1'b1 : rx_meta;
end

typedef enum logic [1:0] {IDLE, START, DATA, STOP} state_t;
state_t state;

logic [$clog2(CLKS_PER_BIT)-1:0] count;
logic [2:0] bit_idx;
logic [7:0] shift;

always_ff @(posedge clk) begin
    if (rst) begin
        state <= IDLE;
        valid <= 0;
        count <= 0;
        bit_idx <= 0;
    end else begin
        valid <= 0;
        case (state)
            IDLE: begin
                count <= 0;
                if (!rx_sync)
                    state <= START;
            end

            // wait half a bit, then check the start bit is still low (not a glitch)
            START: begin
                if (count == CLKS_PER_BIT/2 - 1) begin
                    count <= 0;
                    bit_idx <= 0;
                    state <= rx_sync ? IDLE : DATA;
                end else
                    count <= count + 1;
            end

            DATA: begin
                if (count == CLKS_PER_BIT - 1) begin
                    count <= 0;
                    shift <= {rx_sync, shift[7:1]};
                    bit_idx <= bit_idx + 1;
                    if (bit_idx == 7)
                        state <= STOP;
                end else
                    count <= count + 1;
            end

            STOP: begin
                if (count == CLKS_PER_BIT - 1) begin
                    count <= 0;
                    if (rx_sync) begin
                        data  <= shift;
                        valid <= 1;
                    end
                    state <= IDLE;
                end else
                    count <= count + 1;
            end
        endcase
    end
end

endmodule

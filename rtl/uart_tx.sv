// UART transmitter, 8N1, LSB first. CLKS_PER_BIT = clock frequency / baud rate.
// Pulse start for one cycle with data; busy stays high until the stop bit ends.
module uart_tx #(
    parameter int CLKS_PER_BIT = 868
)(
    input  logic       clk,
    input  logic       rst,
    input  logic       start,
    input  logic [7:0] data,
    output logic       tx,
    output logic       busy
);

logic [$clog2(CLKS_PER_BIT)-1:0] count;
logic [3:0] bit_idx;       // 0 = start bit, 1-8 = data, 9 = stop bit
logic [9:0] frame;

always_ff @(posedge clk) begin
    if (rst) begin
        tx   <= 1;
        busy <= 0;
        count <= 0;
        bit_idx <= 0;
    end else if (!busy) begin
        tx <= 1;
        if (start) begin
            frame   <= {1'b1, data, 1'b0};   // stop, data (LSB first), start
            busy    <= 1;
            count   <= 0;
            bit_idx <= 0;
            tx      <= 0;
        end
    end else begin
        if (count == CLKS_PER_BIT - 1) begin
            count <= 0;
            if (bit_idx == 9)
                busy <= 0;
            else begin
                bit_idx <= bit_idx + 1;
                tx <= frame[bit_idx + 1];
            end
        end else
            count <= count + 1;
    end
end

endmodule

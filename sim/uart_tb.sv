`timescale 1ns/1ps

// Loopback: uart_tx drives uart_rx at the real 100 MHz / 115200 baud setting.
// Sends all 256 byte values back to back and checks each one arrives intact.
module uart_tb;

    localparam int CLKS_PER_BIT = 868;

    logic clk, rst, start, line, busy, valid;
    logic [7:0] tx_data, rx_data;

    uart_tx #(.CLKS_PER_BIT(CLKS_PER_BIT)) tx (
        .clk(clk), .rst(rst), .start(start), .data(tx_data), .tx(line), .busy(busy)
    );

    uart_rx #(.CLKS_PER_BIT(CLKS_PER_BIT)) rx (
        .clk(clk), .rst(rst), .rx(line), .data(rx_data), .valid(valid)
    );

    always #5 clk = ~clk;

    integer errors = 0;
    integer received = 0;

    // checker: bytes must arrive in order 0, 1, 2, ... 255
    always @(posedge clk) begin
        if (valid) begin
            if (rx_data !== received[7:0]) begin
                $display("  MISMATCH byte %0d: got %02h", received, rx_data);
                errors++;
            end
            received++;
        end
    end

    initial begin
        clk = 0; rst = 1; start = 0; tx_data = 0;
        #100 rst = 0;

        for (int b = 0; b < 256; b++) begin
            @(posedge clk);
            while (busy) @(posedge clk);
            tx_data <= b;
            start   <= 1;
            @(posedge clk) start <= 0;
            @(posedge clk);
        end
        while (busy) @(posedge clk);
        repeat (2 * CLKS_PER_BIT) @(posedge clk);

        if (errors == 0 && received == 256)
            $display("UART PASS (256 bytes)");
        else
            $display("UART FAIL: %0d mismatches, %0d of 256 received", errors, received);
        $finish;
    end

endmodule

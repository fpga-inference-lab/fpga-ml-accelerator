`timescale 1ns/1ps

// Full-system test of top.sv through its UART pins, at the real 115200 baud.
// The testbench plays the laptop: it sends each of the 5 test examples as a
// 20-byte request and checks the 14-byte reply against the Python reference.
// It first sends a half request and goes quiet, to check the timeout drops it.
// HEX_DIR is relative to where xsim runs (the repo root by default).
module top_tb;

    parameter HEX_DIR = "model/weights/hex/";

    localparam int CLKS_PER_BIT = 868;
    localparam int TIMEOUT_CLKS = 50_000;   // shortened so the sim stays quick
    localparam int M = 5, K1 = 20, O = 2;

    logic clk, btnC, RsRx, RsTx;
    logic [15:0] led;

    top #(.CLKS_PER_BIT(CLKS_PER_BIT), .TIMEOUT_CLKS(TIMEOUT_CLKS), .HEX_DIR(HEX_DIR)) dut (
        .clk(clk), .btnC(btnC), .RsRx(RsRx), .RsTx(RsTx), .led(led)
    );

    always #5 clk = ~clk;

    logic signed [7:0]  x      [M*K1];
    logic signed [31:0] z2_ref [M*O];

    task automatic send_byte(input logic [7:0] b);
        RsRx = 0;                                   // start bit
        repeat (CLKS_PER_BIT) @(posedge clk);
        for (int i = 0; i < 8; i++) begin
            RsRx = b[i];
            repeat (CLKS_PER_BIT) @(posedge clk);
        end
        RsRx = 1;                                   // stop bit
        repeat (CLKS_PER_BIT) @(posedge clk);
    endtask

    task automatic recv_byte(output logic [7:0] b);
        @(negedge RsTx);                            // start bit begins
        repeat (CLKS_PER_BIT + CLKS_PER_BIT/2) @(posedge clk);
        for (int i = 0; i < 8; i++) begin
            b[i] = RsTx;
            repeat (CLKS_PER_BIT) @(posedge clk);
        end
        if (RsTx !== 1) $display("  bad stop bit");
    endtask

    integer errors = 0;

    initial begin
        logic [7:0]  resp [14];
        logic signed [31:0] got0, got1;
        logic [31:0] cycles;
        logic        expected_pred;

        $readmemh({HEX_DIR, "X_test.hex"},       x);
        $readmemh({HEX_DIR, "z2_reference.hex"}, z2_ref);

        clk = 0; btnC = 1; RsRx = 1;
        repeat (20) @(posedge clk);
        btnC = 0;
        repeat (20) @(posedge clk);

        // half a request, then silence: top must drop it after TIMEOUT_CLKS
        for (int k = 0; k < 7; k++) send_byte(8'h7F);
        if (led[8:4] !== 7) begin
            $display("  expected 7 bytes pending, LEDs show %0d", led[8:4]);
            errors++;
        end
        repeat (TIMEOUT_CLKS + 100) @(posedge clk);
        if (led[8:4] !== 0) begin
            $display("  partial request was not dropped (LEDs show %0d)", led[8:4]);
            errors++;
        end

        for (int m = 0; m < M; m++) begin
            fork
                for (int k = 0; k < K1; k++) send_byte(x[m*K1 + k]);
                for (int i = 0; i < 14; i++) recv_byte(resp[i]);
            join

            got0   = {resp[5],  resp[4],  resp[3],  resp[2]};
            got1   = {resp[9],  resp[8],  resp[7],  resp[6]};
            cycles = {resp[13], resp[12], resp[11], resp[10]};
            expected_pred = z2_ref[m*O + 1] > z2_ref[m*O];

            $display("example %0d: pred %0d  z2 = %0d %0d  (%0d cycles)", m, resp[1], got0, got1, cycles);

            if (resp[0] !== 8'hA5 || resp[1] !== expected_pred ||
                got0 !== z2_ref[m*O] || got1 !== z2_ref[m*O + 1]) begin
                $display("  MISMATCH: expected header a5, pred %0d, z2 = %0d %0d",
                         expected_pred, z2_ref[m*O], z2_ref[m*O + 1]);
                errors++;
            end
            if (led[0] !== expected_pred || led[1] !== 1) begin
                $display("  LED mismatch: led[1:0] = %b", led[1:0]);
                errors++;
            end
        end

        if (errors == 0)
            $display("TOP PASS");
        else
            $display("TOP FAIL: %0d errors", errors);
        $finish;
    end

    initial begin
        #100_000_000;
        $display("TOP FAIL: timeout");
        $finish;
    end

endmodule

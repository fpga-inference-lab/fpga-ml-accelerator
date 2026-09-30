`timescale 1ns/1ps

// Checks dense.sv as both network layers against the Python golden reference
// (model/export_reference.py). Each layer is tested on its own:
//   layer 1: X_test (5x20) @ W1 + b1, ReLU      -> z1_reference, a1_q
//   layer 2: a1_q   (5x16) @ W2 + b2, no ReLU   -> z2_reference
// HEX_DIR is relative to where xsim runs (the repo root by default).
module dense_tb;

    parameter HEX_DIR = "model/weights/hex/";

    localparam int M = 5, K1 = 20, H = 16, O = 2;

    logic clk, rst, start1, start2, done1, done2;

    logic signed [7:0]  x      [M*K1];
    logic signed [7:0]  a1_ref [M*H];
    logic signed [31:0] z1_ref [M*H];
    logic signed [31:0] z2_ref [M*O];

    logic signed [31:0] y1   [M*H];
    logic signed [7:0]  y1_q [M*H];
    logic signed [31:0] y2   [M*O];
    logic signed [7:0]  y2_q [M*O];

    dense #(.M(M), .K(K1), .COLS(H), .APPLY_ACT(1),
            .W_FILE({HEX_DIR, "W1.hex"}), .B_FILE({HEX_DIR, "b1.hex"})) layer1 (
        .clk(clk), .rst(rst), .start(start1), .x(x), .y(y1), .y_q(y1_q), .done(done1)
    );

    dense #(.M(M), .K(H), .COLS(O), .APPLY_ACT(0),
            .W_FILE({HEX_DIR, "W2.hex"}), .B_FILE({HEX_DIR, "b2.hex"})) layer2 (
        .clk(clk), .rst(rst), .start(start2), .x(a1_ref), .y(y2), .y_q(y2_q), .done(done2)
    );

    always #5 clk = ~clk;

    integer errors = 0;
    integer cycles;

    initial begin
        $readmemh({HEX_DIR, "X_test.hex"},       x);
        $readmemh({HEX_DIR, "a1_q.hex"},         a1_ref);
        $readmemh({HEX_DIR, "z1_reference.hex"}, z1_ref);
        $readmemh({HEX_DIR, "z2_reference.hex"}, z2_ref);

        clk = 0; rst = 1; start1 = 0; start2 = 0;
        #20 rst = 0;

        // layer 1
        @(posedge clk) start1 <= 1;
        @(posedge clk) start1 <= 0;
        cycles = 1;
        while (!done1) begin
            @(posedge clk);
            cycles++;
        end
        $display("layer 1 done in %0d cycles", cycles);

        for (int i = 0; i < M*H; i++) begin
            if (y1[i] !== z1_ref[i] || y1_q[i] !== a1_ref[i]) begin
                $display("  MISMATCH layer1 [%0d][%0d]: got z=%0d q=%0d, expected z=%0d q=%0d",
                         i / H, i % H, y1[i], y1_q[i], z1_ref[i], a1_ref[i]);
                errors++;
            end
        end

        // layer 2
        @(posedge clk) start2 <= 1;
        @(posedge clk) start2 <= 0;
        cycles = 1;
        while (!done2) begin
            @(posedge clk);
            cycles++;
        end
        $display("layer 2 done in %0d cycles", cycles);

        for (int i = 0; i < M*O; i++) begin
            if (y2[i] !== z2_ref[i]) begin
                $display("  MISMATCH layer2 [%0d][%0d]: got %0d, expected %0d",
                         i / O, i % O, y2[i], z2_ref[i]);
                errors++;
            end
        end

        $display("z2:");
        for (int r = 0; r < M; r++)
            $display("  %0d %0d", y2[r*O], y2[r*O + 1]);

        if (errors == 0)
            $display("DENSE PASS (%0d layer-1 values, %0d layer-2 values)", M*H, M*O);
        else
            $display("DENSE FAIL: %0d mismatches", errors);
        $finish;
    end

    initial begin
        #200000;
        $display("DENSE FAIL: timeout");
        $finish;
    end

endmodule

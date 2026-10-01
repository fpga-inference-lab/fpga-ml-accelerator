`timescale 1ns/1ps

// Runs the 5 test examples through fast_mlp one at a time and checks z2 and the
// prediction against the Python golden reference. Also reports the latency.
// HEX_DIR is relative to where xsim runs (the repo root by default).
module fast_mlp_tb;

    parameter HEX_DIR = "model/weights/hex/";
    parameter int TREE_REG_EVERY = 3;

    localparam int M = 5, K1 = 20, O = 2;

    logic clk, rst, start, done;
    logic signed [7:0]  x      [K1];
    logic signed [31:0] z2     [O];
    logic               pred   [1];
    logic signed [7:0]  x_all  [M*K1];
    logic signed [31:0] z2_ref [M*O];

    fast_mlp #(.K1(K1), .H(16), .O(O), .TREE_REG_EVERY(TREE_REG_EVERY),
               .W1_FILE({HEX_DIR, "W1.hex"}), .B1_FILE({HEX_DIR, "b1.hex"}),
               .W2_FILE({HEX_DIR, "W2.hex"}), .B2_FILE({HEX_DIR, "b2.hex"})) dut (
        .clk(clk), .rst(rst), .start(start), .x(x), .z2(z2), .pred(pred), .done(done)
    );

    always #5 clk = ~clk;

    integer errors = 0;
    integer cycles;
    logic   expected_pred;

    initial begin
        $readmemh({HEX_DIR, "X_test.hex"},       x_all);
        $readmemh({HEX_DIR, "z2_reference.hex"}, z2_ref);

        clk = 0; rst = 1; start = 0;
        #20 rst = 0;

        for (int m = 0; m < M; m++) begin
            @(posedge clk);
            for (int k = 0; k < K1; k++) x[k] <= x_all[m*K1 + k];
            start <= 1;
            @(posedge clk) start <= 0;
            // done from the previous example clears on this edge, so step past it first
            cycles = 1;
            do begin
                @(posedge clk);
                cycles++;
            end while (!done);

            expected_pred = z2_ref[m*O + 1] > z2_ref[m*O];
            $display("example %0d: pred %0d  z2 = %0d %0d  (%0d cycles)", m, pred[0], z2[0], z2[1], cycles);
            if (z2[0] !== z2_ref[m*O] || z2[1] !== z2_ref[m*O + 1] || pred[0] !== expected_pred) begin
                $display("  MISMATCH: expected pred %0d  z2 = %0d %0d",
                         expected_pred, z2_ref[m*O], z2_ref[m*O + 1]);
                errors++;
            end
        end

        if (errors == 0)
            $display("FAST MLP PASS");
        else
            $display("FAST MLP FAIL: %0d mismatches", errors);
        $finish;
    end

    initial begin
        #100000;
        $display("FAST MLP FAIL: timeout");
        $finish;
    end

endmodule

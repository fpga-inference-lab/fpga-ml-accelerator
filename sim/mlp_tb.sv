`timescale 1ns/1ps

// Runs the full two-layer MLP on the 5 test examples and checks z2 and the
// predictions against the Python golden reference (model/export_reference.py).
// Inference is run twice to make sure the top restarts cleanly.
// HEX_DIR is relative to where xsim runs (the repo root by default).
module mlp_tb;

    parameter HEX_DIR = "model/weights/hex/";

    localparam int M = 5, K1 = 20, H = 16, O = 2;

    logic clk, rst, start, done;
    logic signed [7:0]  x      [M*K1];
    logic signed [31:0] z2     [M*O];
    logic               pred   [M];
    logic signed [31:0] z2_ref [M*O];

    mlp #(.M(M), .K1(K1), .H(H), .O(O),
          .W1_FILE({HEX_DIR, "W1.hex"}), .B1_FILE({HEX_DIR, "b1.hex"}),
          .W2_FILE({HEX_DIR, "W2.hex"}), .B2_FILE({HEX_DIR, "b2.hex"})) dut (
        .clk(clk), .rst(rst), .start(start), .x(x), .z2(z2), .pred(pred), .done(done)
    );

    always #5 clk = ~clk;

    integer errors = 0;
    integer cycles;
    logic   expected_pred;

    task automatic run_and_check(input int run);
        @(posedge clk) start <= 1;
        @(posedge clk) start <= 0;
        // done from the previous run clears on this edge, so step past it before polling
        cycles = 1;
        do begin
            @(posedge clk);
            cycles++;
        end while (!done);
        $display("run %0d: inference done in %0d cycles", run, cycles);

        for (int m = 0; m < M; m++) begin
            expected_pred = z2_ref[m*O + 1] > z2_ref[m*O];
            for (int o = 0; o < O; o++)
                if (z2[m*O + o] !== z2_ref[m*O + o]) begin
                    $display("  MISMATCH z2[%0d][%0d]: got %0d, expected %0d",
                             m, o, z2[m*O + o], z2_ref[m*O + o]);
                    errors++;
                end
            if (pred[m] !== expected_pred) begin
                $display("  MISMATCH pred[%0d]: got %0d, expected %0d", m, pred[m], expected_pred);
                errors++;
            end
        end
    endtask

    initial begin
        $readmemh({HEX_DIR, "X_test.hex"},       x);
        $readmemh({HEX_DIR, "z2_reference.hex"}, z2_ref);

        clk = 0; rst = 1; start = 0;
        #20 rst = 0;

        run_and_check(1);
        run_and_check(2);

        $display("predictions: %0d %0d %0d %0d %0d", pred[0], pred[1], pred[2], pred[3], pred[4]);
        if (errors == 0)
            $display("MLP PASS");
        else
            $display("MLP FAIL: %0d mismatches", errors);
        $finish;
    end

    initial begin
        #200000;
        $display("MLP FAIL: timeout");
        $finish;
    end

endmodule

`timescale 1ns/1ps

// Checks systolic_array.sv against the Python golden matrix products
// (model/export_unit_vectors.py). Each case is a 4x4 A and B and the expected
// C = A @ B (int32), loaded from hex. Cases: zeros, identity on either side, mixed
// signs, extreme values (-128 and 127), B all ones (the original hand test), two tiles
// of the real X_test @ W1, and seeded random.
//
// For each case the testbench resets the array, checks every c[i][j] is 0, then plays
// the skewed schedule the array needs: row i of A is delayed i cycles and column j of
// B is delayed j cycles, so A[i][k] and B[k][j] meet at PE(i,j) on cycle i+j+k. After
// the last product lands it checks all 16 results, then runs two more cycles of zero
// input and checks again, to show the results are stable once the data has drained.
// Odd-numbered cases also freeze the array (en low) for 3 cycles mid-feed while the
// inputs carry extreme garbage; the results must not change.
//
// Inputs change on the falling edge and outputs are read on the next falling edge, so
// the testbench never races the rising edge the design samples on.
// HEX_DIR is relative to where xsim runs (the repo root by default).
module systolic_array_tb;

    parameter HEX_DIR = "model/weights/hex/";
    parameter int N = 4;

    localparam int MAXCASES = 64;

    logic               clk;
    logic               rst;
    logic               en;
    logic signed [7:0]  a_west  [N];
    logic signed [7:0]  b_north [N];
    logic signed [31:0] c       [N][N];

    systolic_array #(.N(N)) dut (
        .clk(clk),
        .rst(rst),
        .en(en),
        .a_west(a_west),
        .b_north(b_north),
        .c(c)
    );

    always #5 clk = ~clk;

    logic [31:0]        n_mem [1];
    logic signed [7:0]  A     [MAXCASES*N*N];
    logic signed [7:0]  B     [MAXCASES*N*N];
    logic signed [31:0] C_exp [MAXCASES*N*N];

    integer errors = 0;
    integer ncases;

    // drive the inputs for schedule step t of case cs (zeros outside the data window)
    task automatic drive(input int cs, input int t);
        for (int i = 0; i < N; i++) begin
            a_west[i]  = ((t - i) >= 0 && (t - i) < N) ? A[cs*N*N + i*N + (t - i)] : 8'sd0;
            b_north[i] = ((t - i) >= 0 && (t - i) < N) ? B[cs*N*N + (t - i)*N + i] : 8'sd0;
        end
    endtask

    task automatic check_c(input int cs, input string when_);
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                if (c[i][j] !== C_exp[cs*N*N + i*N + j]) begin
                    if (errors < 10)
                        $display("  MISMATCH case %0d (%s) c[%0d][%0d]: got %0d, expected %0d",
                                 cs, when_, i, j, c[i][j], C_exp[cs*N*N + i*N + j]);
                    errors++;
                end
    endtask

    task automatic run_case(input int cs);
        bit stall = (cs % 2 == 1);

        // reset (with en high, so this also shows reset wins), then every c must be 0
        @(negedge clk);
        rst = 1; en = 1;
        for (int i = 0; i < N; i++) begin a_west[i] = 0; b_north[i] = 0; end
        @(negedge clk);
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                if (c[i][j] !== 32'sd0) begin
                    if (errors < 10)
                        $display("  MISMATCH case %0d (after reset) c[%0d][%0d] = %0d, expected 0", cs, i, j, c[i][j]);
                    errors++;
                end
        rst = 0;

        // skewed feed: step t is consumed on the next rising edge
        for (int t = 0; t <= 3*N - 3; t++) begin
            if (stall && t == 4) begin
                repeat (3) begin
                    en = 0;
                    for (int i = 0; i < N; i++) begin
                        a_west[i]  = -8'sd128;
                        b_north[i] = 8'sd127;
                    end
                    @(negedge clk);
                end
                en = 1;
            end
            drive(cs, t);
            @(negedge clk);
        end

        check_c(cs, "after last product");

        // drain: zero input must leave the results alone
        for (int i = 0; i < N; i++) begin a_west[i] = 0; b_north[i] = 0; end
        repeat (2) @(negedge clk);
        check_c(cs, "after drain");
    endtask

    initial begin
        $readmemh({HEX_DIR, "arr_n.hex"}, n_mem);
        ncases = n_mem[0];

        if (ncases < 1 || ncases > MAXCASES) begin
            $display("ARRAY FAIL: bad case count %0d in arr_n.hex (MAXCASES = %0d)", ncases, MAXCASES);
            $finish;
        end

        $readmemh({HEX_DIR, "arr_A.hex"}, A, 0, ncases*N*N - 1);
        $readmemh({HEX_DIR, "arr_B.hex"}, B, 0, ncases*N*N - 1);
        $readmemh({HEX_DIR, "arr_C.hex"}, C_exp, 0, ncases*N*N - 1);


        clk = 0; rst = 1; en = 0;
        for (int i = 0; i < N; i++) begin a_west[i] = 0; b_north[i] = 0; end

        for (int cs = 0; cs < ncases; cs++)
            run_case(cs);

        if (errors == 0)
            $display("ARRAY PASS (%0d cases)", ncases);
        else
            $display("ARRAY FAIL: %0d mismatches", errors);
        $finish;
    end

    initial begin
        #10_000_000;
        $display("ARRAY FAIL: timeout");
        $finish;
    end

endmodule

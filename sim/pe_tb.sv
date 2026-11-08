`timescale 1ns/1ps

// Checks pe.sv cycle by cycle against the Python golden vectors
// (model/export_unit_vectors.py), the same cycle-by-cycle stimulus mac_unit_tb uses.
// Each line of the vector files is one clock cycle: the testbench drives rst, en, a_in,
// b_in, then after the clock edge compares all three outputs with the expected values:
//   acc    running sum of a_in * b_in
//   a_out  a_in passed along to the east neighbor
//   b_out  b_in passed along to the south neighbor
// The forwarding outputs are what make the array systolic, so this checks the part
// mac_unit_tb cannot: a_out/b_out follow the inputs only when en is high, hold when it
// is low, and clear on reset. The vectors include enable low with live inputs, reset in
// the middle of a run, 80 real dot products (X_test x W1) and seeded random data with
// rst/en toggling.
// Inputs change on the falling edge and outputs are read on the next falling edge, so
// the testbench never races the rising edge the design samples on.
// HEX_DIR is relative to where xsim runs (the repo root by default).
module pe_tb;

    parameter HEX_DIR = "model/weights/hex/";

    localparam int MAXN = 4096;

    logic               clk;
    logic               rst;
    logic               en;
    logic signed [7:0]  a_in;
    logic signed [7:0]  b_in;
    logic signed [7:0]  a_out;
    logic signed [7:0]  b_out;
    logic signed [31:0] acc;

    pe dut (
        .clk(clk),
        .rst(rst),
        .en(en),
        .a_in(a_in),
        .b_in(b_in),
        .a_out(a_out),
        .b_out(b_out),
        .acc(acc)
    );

    always #5 clk = ~clk;

    logic [31:0]        n_mem    [1];
    logic [3:0]         rst_v    [MAXN];
    logic [3:0]         en_v     [MAXN];
    logic signed [7:0]  a_v      [MAXN];
    logic signed [7:0]  b_v      [MAXN];
    logic signed [31:0] acc_exp  [MAXN];
    logic signed [7:0]  aout_exp [MAXN];
    logic signed [7:0]  bout_exp [MAXN];

    integer errors = 0;
    integer n;

    initial begin
        $readmemh({HEX_DIR, "unit_n.hex"},    n_mem);
        n = n_mem[0];

        if (n < 1 || n > MAXN) begin
            $display("PE FAIL: bad cycle count %0d in unit_n.hex (MAXN = %0d)", n, MAXN);
            $finish;
        end

        $readmemh({HEX_DIR, "unit_rst.hex"},  rst_v, 0, n - 1);
        $readmemh({HEX_DIR, "unit_en.hex"},   en_v, 0, n - 1);
        $readmemh({HEX_DIR, "unit_a.hex"},    a_v, 0, n - 1);
        $readmemh({HEX_DIR, "unit_b.hex"},    b_v, 0, n - 1);
        $readmemh({HEX_DIR, "unit_acc.hex"},  acc_exp, 0, n - 1);
        $readmemh({HEX_DIR, "unit_aout.hex"}, aout_exp, 0, n - 1);
        $readmemh({HEX_DIR, "unit_bout.hex"}, bout_exp, 0, n - 1);

        clk = 0; rst = 0; en = 0; a_in = 0; b_in = 0;

        // iteration i drives cycle i and checks the result of cycle i-1
        for (int i = 0; i <= n; i++) begin
            @(negedge clk);
            if (i > 0 && (acc !== acc_exp[i-1] || a_out !== aout_exp[i-1] || b_out !== bout_exp[i-1])) begin
                if (errors < 10)
                    $display("  MISMATCH cycle %0d: got acc=%0d a_out=%0d b_out=%0d, expected %0d %0d %0d",
                             i-1, acc, a_out, b_out, acc_exp[i-1], aout_exp[i-1], bout_exp[i-1]);
                errors++;
            end
            if (i < n) begin
                rst  = rst_v[i][0];
                en   = en_v[i][0];
                a_in = a_v[i];
                b_in = b_v[i];
            end
        end

        if (errors == 0)
            $display("PE PASS (%0d cycles)", n);
        else
            $display("PE FAIL: %0d mismatches", errors);
        $finish;
    end

    initial begin
        #1_000_000;
        $display("PE FAIL: timeout");
        $finish;
    end

endmodule

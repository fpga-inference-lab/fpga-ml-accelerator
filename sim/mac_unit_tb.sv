`timescale 1ns/1ps

// Checks mac_unit.sv cycle by cycle against the Python golden vectors
// (model/export_unit_vectors.py). Each line of the vector files is one clock cycle:
// the testbench drives rst, en, a, b, then after the clock edge compares acc with the
// expected value on that line. The vectors cover zeros, extreme values, mixed signs,
// enable low with live inputs, reset in the middle of a run, 80 real dot products
// (X_test x W1) and seeded random data. The Python script checks its own cycle model
// against NumPy dot products before writing the files.
// Inputs change on the falling edge and outputs are read on the next falling edge, so
// the testbench never races the rising edge the design samples on.
// HEX_DIR is relative to where xsim runs (the repo root by default).
module mac_unit_tb;

    parameter HEX_DIR = "model/weights/hex/";

    localparam int MAXN = 4096;

    logic               clk;
    logic               rst;
    logic               en;
    logic signed [7:0]  a;
    logic signed [7:0]  b;
    logic signed [31:0] acc;

    mac_unit dut (
        .clk(clk),
        .rst(rst),
        .en(en),
        .a(a),
        .b(b),
        .acc(acc)
    );

    always #5 clk = ~clk;

    logic [31:0]        n_mem   [1];
    logic [3:0]         rst_v   [MAXN];
    logic [3:0]         en_v    [MAXN];
    logic signed [7:0]  a_v     [MAXN];
    logic signed [7:0]  b_v     [MAXN];
    logic signed [31:0] acc_exp [MAXN];

    integer errors = 0;
    integer n;

    initial begin
        $readmemh({HEX_DIR, "unit_n.hex"},   n_mem);
        n = n_mem[0];

        if (n < 1 || n > MAXN) begin
            $display("MAC FAIL: bad cycle count %0d in unit_n.hex (MAXN = %0d)", n, MAXN);
            $finish;
        end

        $readmemh({HEX_DIR, "unit_rst.hex"}, rst_v, 0, n - 1);
        $readmemh({HEX_DIR, "unit_en.hex"},  en_v, 0, n - 1);
        $readmemh({HEX_DIR, "unit_a.hex"},   a_v, 0, n - 1);
        $readmemh({HEX_DIR, "unit_b.hex"},   b_v, 0, n - 1);
        $readmemh({HEX_DIR, "unit_acc.hex"}, acc_exp, 0, n - 1);

        clk = 0; rst = 0; en = 0; a = 0; b = 0;

        // iteration i drives cycle i and checks the result of cycle i-1
        for (int i = 0; i <= n; i++) begin
            @(negedge clk);
            if (i > 0 && acc !== acc_exp[i-1]) begin
                if (errors < 10)
                    $display("  MISMATCH cycle %0d: got acc=%0d, expected %0d", i-1, acc, acc_exp[i-1]);
                errors++;
            end
            if (i < n) begin
                rst = rst_v[i][0];
                en  = en_v[i][0];
                a   = a_v[i];
                b   = b_v[i];
            end
        end

        if (errors == 0)
            $display("MAC PASS (%0d cycles)", n);
        else
            $display("MAC FAIL: %0d mismatches", errors);
        $finish;
    end

    initial begin
        #1_000_000;
        $display("MAC FAIL: timeout");
        $finish;
    end

endmodule

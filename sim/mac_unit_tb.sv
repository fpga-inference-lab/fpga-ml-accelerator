`timescale 1ns/1ps

module mac_unit_tb;

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

    initial begin
        clk = 0;
        rst = 1;
        en  = 0;
        a   = 0;
        b   = 0;

        #10 rst = 0;

        #10 a = 2; b = 3; en = 1;
        #10 a = 1; b = 4;
        #10 a = 5; b = 1;
        #10 en = 0;

        #20;
        $display("Accumulator = %d (expected 15)", acc);
        $finish;
    end

endmodule
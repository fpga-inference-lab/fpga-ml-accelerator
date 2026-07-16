`timescale 1ns/1ps

module systolic_array_tb;

    parameter N = 4;

    logic               clk;
    logic               rst;
    logic               en;
    logic signed [7:0]  a_west [N];
    logic signed [7:0]  b_north [N];
    logic signed [31:0] c [N][N];

    systolic_array #(.N(N)) dut (
        .clk(clk),
        .rst(rst),
        .en(en),
        .a_west(a_west),
        .b_north(b_north),
        .c(c)
    );

    always #5 clk = ~clk;

    logic signed [7:0] A [N][N];
    logic signed [7:0] B [N][N];

    integer cycle;
    integer i, j;

    initial begin
        A[0][0]=2; A[0][1]=3; A[0][2]=1; A[0][3]=0;
        A[1][0]=1; A[1][1]=4; A[1][2]=2; A[1][3]=3;
        A[2][0]=0; A[2][1]=2; A[2][2]=5; A[2][3]=1;
        A[3][0]=3; A[3][1]=1; A[3][2]=0; A[3][3]=4;

        for (i=0; i<N; i++)
            for (j=0; j<N; j++)
                B[i][j] = 1;

        clk = 0;
        rst = 1;
        en  = 0;
        for (i=0; i<N; i++) begin
            a_west[i]  = 0;
            b_north[i] = 0;
        end

        #10 rst = 0;
        en = 1;

        for (cycle = 0; cycle < N + N + N; cycle++) begin
            @(posedge clk);
            for (i = 0; i < N; i++) begin
                if ((cycle - i) >= 0 && (cycle - i) < N)
                    a_west[i] = A[i][cycle - i];
                else
                    a_west[i] = 0;

                if ((cycle - i) >= 0 && (cycle - i) < N)
                    b_north[i] = B[cycle - i][i];
                else
                    b_north[i] = 0;
            end
        end

        en = 0;
        #50;

        $display("Result matrix C:");
        for (i = 0; i < N; i++) begin
            $display("Row %0d: %0d %0d %0d %0d", i, c[i][0], c[i][1], c[i][2], c[i][3]);
        end
        $display("Expected row sums: 6, 10, 8, 8");

        $finish;
    end

endmodule
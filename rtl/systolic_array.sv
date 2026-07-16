module systolic_array #(
    parameter int N = 4
)(
    input logic clk,
    input logic rst,
    input logic en,
    input logic signed [7:0] a_west [N],
    input logic signed [7:0] b_north [N],
    output logic signed [31:0] c [N][N]
);

logic signed [7:0] a_wire [N][N+1];
logic signed [7:0] b_wire [N+1][N];

genvar i, j;
generate
    for(i = 0; i<N; i++) begin : row
        for(j = 0; j<N; j++) begin : col
            pe pe_inst(
                .clk (clk),
                .rst (rst),
                .en (en),
                .a_in (a_wire[i][j]),
                .b_in (b_wire[i][j]),
                .a_out (a_wire[i][j+1]),    
                .b_out (b_wire[i+1][j]),    
                .acc   (c[i][j])            
                );
            end
        end
    endgenerate

    
    generate
        for (i = 0; i < N; i++) assign a_wire[i][0] = a_west[i];
        for (j = 0; j < N; j++) assign b_wire[0][j] = b_north[j];
    endgenerate
endmodule
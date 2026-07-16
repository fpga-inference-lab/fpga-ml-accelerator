module mac_unit(
   input logic clk,
   input logic rst,
   input logic en,
   input logic signed [7:0] a,
   input logic signed [7:0] b,
   output logic signed [31:0] acc
    );
    
    always_ff @(posedge clk) begin
        if(rst)
            acc <=0;
        else if (en)
            acc <= acc + a * b;
    end

endmodule
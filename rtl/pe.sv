// use_dsp: Vivado builds small (8x8) multiplies from LUTs by default; this puts
// the multiply and the accumulator inside one DSP48 slice instead.
(* use_dsp = "yes" *)
module pe(
    input logic clk,
    input logic rst,
    input logic en,
    input logic signed [7:0] a_in,
    input logic signed [7:0] b_in,
    output logic signed [7:0] a_out,
    output logic signed [7:0] b_out,
    output logic signed [31:0] acc 
);

always_ff @(posedge clk) begin
        if(rst) begin
            acc <= 0;
            a_out <=0;
            b_out <= 0;
        end else if (en) begin
            acc <= acc + a_in * b_in;
            a_out <= a_in;
            b_out <= b_in;
        end
    end
endmodule
module top(
input [7:0]a1,
input [7:0]a2,
input clk,
input reset,
output reg [15:0]pro
);
reg[7:0]A1;
reg[7:0]A2;
wire[15:0]PRO;
APPROX dut (
.N1(A1),
.N2(A2),
.P_APPROX(PRO)
);
always @(posedge clk)
begin
    A1  = a1;
    A2  = a2;
    pro = PRO;
end
endmodule

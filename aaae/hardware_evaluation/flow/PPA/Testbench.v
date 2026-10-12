`timescale 1ns / 1ps
module Testbench();
parameter TIME_PERIOD = 5;
integer fid_N1;
integer fid_N2;
integer fid_P_APPROX;
integer i,j,k;
reg clk;
reg reset;
reg [7:0]A;
reg [7:0]B;
wire [15:0]pro;
initial
begin
reset = 0;
clk = 0;
#TIME_PERIOD
reset = 1;
end
always #2.5 clk = ~clk;
always @(posedge clk or negedge reset)
begin
if(!reset)
begin
A <= 0;
B <= 0;
end
else
begin
A <= #0.05 {$random} % 256;
B <= #0.05 {$random} % 256;
end
end 
top u0
(
.a1(A),
.a2(B),
.clk(clk),
.reset(reset),
.pro(pro)
);
initial
begin
fid_N1 = $fopen("N1.txt","w");
for(i = 0; i < 200; i = i + 1)
begin
#TIME_PERIOD $fdisplay(fid_N1,"%d",A);
end
$fclose(fid_N1);
end
initial
begin
fid_N2 = $fopen("N2.txt","w");
for(j = 0; j < 200; j = j + 1)
begin
#TIME_PERIOD $fdisplay(fid_N2,"%d",B);
end
$fclose(fid_N2);
end
initial
begin
fid_P_APPROX = $fopen("P_APPROX.txt","w");
for(k = 0; k < 202; k = k + 1)
begin
#TIME_PERIOD $fdisplay(fid_P_APPROX,"%d",pro);
end
$fclose(fid_P_APPROX);
$finish;
end
initial begin
$sdf_annotate("./timing.sdf",top,,,"MAXIMUM","1:1:1","FROM_MTM");
end
initial begin
    $vcdpluson;
end
initial begin
$dumpfile("top.vcd");
$dumpvars;
end
endmodule

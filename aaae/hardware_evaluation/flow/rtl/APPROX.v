`timescale 1ns / 1ps
module APPROX(
input [7:0] N1,
input [7:0] N2,
output [15:0] P_APPROX
);
wire pp_0;
wire [1:0] pp_1;
wire [2:0] pp_2;
wire [3:0] pp_3; 
wire [3:0] pp_4;
wire [3:0] pp_5;
wire [3:0] pp_6;
wire [3:0] pp_7;
wire [3:0] pp_8;
wire [3:0] pp_9;
wire [3:0] pp_10;
wire [3:0] pp_11;
wire [3:0] pp_12;
wire [1:0] pp_13;
wire pp_14; 
STAGE1 STAGE1
(
.A(N1),
.B(N2),
.pp_0(pp_0),
.pp_1(pp_1),
.pp_2(pp_2),
.pp_3(pp_3), 
.pp_4(pp_4),
.pp_5(pp_5),
.pp_6(pp_6),
.pp_7(pp_7),
.pp_8(pp_8),
.pp_9(pp_9),
.pp_10(pp_10),
.pp_11(pp_11),
.pp_12(pp_12),
.pp_13(pp_13),
.pp_14(pp_14)
);
STAGE2 STAGE2
(
.pp_0(pp_0),
.pp_1(pp_1),
.pp_2(pp_2),
.pp_3(pp_3), 
.pp_4(pp_4),
.pp_5(pp_5),
.pp_6(pp_6),
.pp_7(pp_7),
.pp_8(pp_8),
.pp_9(pp_9),
.pp_10(pp_10),
.pp_11(pp_11),
.pp_12(pp_12),
.pp_13(pp_13),
.pp_14(pp_14),
.approx(P_APPROX)
);
endmodule

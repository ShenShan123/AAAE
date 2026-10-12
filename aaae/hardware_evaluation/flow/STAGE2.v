`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2023/02/23 15:47:43
// Design Name: 
// Module Name: STAGE2
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module STAGE2(
 input pp_0,
 input clk,
 input [1:0] pp_1,
 input [2:0] pp_2,
 input [3:0] pp_3, 
 input [3:0] pp_4,
 input [3:0] pp_5,
 input [3:0] pp_6,
 input [3:0] pp_7,
 input [3:0] pp_8,
 input [3:0] pp_9,
 input [3:0] pp_10,
 input [3:0] pp_11,
 input [3:0] pp_12,
 input [1:0] pp_13,
 input  pp_14,
 output [15:0] approx 
    );
    wire [14:0] A;
    wire [14:0] B;
   

    
    wire pp_2_exa_s;
    wire pp_2_exa_c;
    wire pp_3_pro_s;
    wire pp_3_pro_c;
    wire pp_4_pro_s;
    wire pp_4_pro_c;
    wire pp_5_pro_s;
    wire pp_5_pro_c;
    wire pp_6_pro_s;
    wire pp_6_pro_c;
    wire pp_7_pro_s;
    wire pp_7_pro_c;
          
    wire pp_8_exa_c1;
    wire pp_8_exa_c2;
    wire pp_8_exa_s;   
    wire pp_9_exa_c1;
    wire pp_9_exa_c2;
    wire pp_9_exa_s;  
    wire pp_10_exa_c1;
    wire pp_10_exa_c2;
    wire pp_10_exa_s;
    wire pp_11_exa_c1;
    wire pp_11_exa_c2;
    wire pp_11_exa_s;
    wire pp_12_exa_c1;
    wire pp_12_exa_c2;
    wire pp_12_exa_s;
    wire pp_13_exa_s;
    wire pp_13_exa_c;
    wire gnd_const;

    LOGIC0_X1 u_logic0_stage2 (
        .Z(gnd_const)
    );
    
    
    assign A[0] = pp_0;
    assign B[0] = gnd_const;
    assign A[1] = pp_1[0];
    assign B[1] = pp_1[1];
    assign A[2] = pp_2_exa_s;
    assign B[2] = pp_2[2];        
    assign A[3] = pp_3_pro_s;
    assign B[3] = pp_2_exa_c;  
    assign A[4] = pp_4_pro_s;
    assign B[4] = pp_3_pro_c;    
    assign A[5] = pp_5_pro_s;
    assign B[5] = pp_4_pro_c;
    assign A[6] = pp_6_pro_s;
    assign B[6] = pp_5_pro_c;
    assign A[7] = pp_7_pro_s;
    assign B[7] = pp_6_pro_c;  

          
    assign A[8] = pp_8_exa_s;
    assign B[8] = pp_7_pro_c;
    assign A[9] = pp_9_exa_s;
    assign B[9] = pp_8_exa_c1;
        
    assign A[10] = pp_10_exa_s;
    assign B[10] = pp_9_exa_c1;
    assign A[11] = pp_11_exa_s;
    assign B[11] = pp_10_exa_c1;
    assign A[12] = pp_12_exa_s;
    assign B[12] = pp_11_exa_c1;
    
    assign A[13] = pp_13_exa_s;
    assign B[13] = pp_12_exa_c1;
    assign A[14] = pp_14;
    assign B[14] = pp_13_exa_c;
    
    //assign approx = A + B;
    
    ///

   
    Half_adder Half_adder_pp_2
    (
        .a(pp_2[0]),
        .b(pp_2[1]),
        .S(pp_2_exa_s),
        .C(pp_2_exa_c)
    );
    
    ///
    AP1 AP1_pp_3
    (
        .a(pp_3[0]),
        .b(pp_3[1]),
        .c(pp_3[2]),
        .d(pp_3[3]),
        .S(pp_3_pro_s),
        .C(pp_3_pro_c)
    );
    //
    AP1 AP1_pp_4
    (
        .a(pp_4[0]),
        .b(pp_4[1]),
        .c(pp_4[2]),
        .d(pp_4[3]),
        .S(pp_4_pro_s),
        .C(pp_4_pro_c)
    );
    //
    AP1 AP1_pp_5
    (
        .a(pp_5[0]),
        .b(pp_5[1]),
        .c(pp_5[2]),
        .d(pp_5[3]),
        .S(pp_5_pro_s),
        .C(pp_5_pro_c)
    );   
    //
    
    AP1 AP1_pp_6
    (
        .a(pp_6[0]),
        .b(pp_6[1]),
        .c(pp_6[2]),
        .d(pp_6[3]),
        .S(pp_6_pro_s),
        .C(pp_6_pro_c)
    );  
//
    AP1 AP1_pp_7
    (
        .a(pp_7[0]),
        .b(pp_7[1]),
        .c(pp_7[2]),
        .d(pp_7[3]),
        .S(pp_7_pro_s),
        .C(pp_7_pro_c)
    );  
//
    E_4_2 Exact_42_pp_8
    (
        .a(pp_8[0]),
        .b(pp_8[1]),
        .c(pp_8[2]),
        .d(pp_8[3]),
        .CIN(gnd_const),
        .COUT(pp_8_exa_c2),
        .S(pp_8_exa_s),
        .C(pp_8_exa_c1)
    );  
    //
    E_4_2 Exact_42_pp_9
    (
        .a(pp_9[0]),
        .b(pp_9[1]),
        .c(pp_9[2]),
        .d(pp_9[3]),
        .CIN(pp_8_exa_c2),
        .COUT(pp_9_exa_c2),
        .S(pp_9_exa_s),
        .C(pp_9_exa_c1)
    );
    //
    E_4_2 Exact_42_pp_10
    (
        .a(pp_10[0]),
        .b(pp_10[1]),
        .c(pp_10[2]),
        .d(pp_10[3]),
        .CIN(pp_9_exa_c2),
        .COUT(pp_10_exa_c2),
        .S(pp_10_exa_s),
        .C(pp_10_exa_c1)
    );  
    //
    E_4_2 Exact_42_pp_11
    (
        .a(pp_11[0]),
        .b(pp_11[1]),
        .c(pp_11[2]),
        .d(pp_11[3]),
        .CIN(pp_10_exa_c2),
        .COUT(pp_11_exa_c2),
        .S(pp_11_exa_s),
        .C(pp_11_exa_c1)
    );
//
    E_4_2 Exact_42_pp_12
    (
        .a(pp_12[0]),
        .b(pp_12[1]),
        .c(pp_12[2]),
        .d(pp_12[3]),
        .CIN(pp_11_exa_c2),
        .COUT(pp_12_exa_c2),
        .S(pp_12_exa_s),
        .C(pp_12_exa_c1)
    );
    //
    Full_adder Full_adder_pp_13
    (
        .a(pp_13[0]),
        .b(pp_13[1]),
        .CIN(pp_12_exa_c2),
        .S(pp_13_exa_s),
        .C(pp_13_exa_c)
    );
	adder_15bit adder_15bit(
	.clk(clk),
	.a(A),
	.b(B),
	.sum(approx)
	);
    
`ifdef SIM
integer Half_adder_pp_2_2;

integer AP1_pp_3_2;

integer AP1_pp_4_2;

integer AP1_pp_5_2;

integer AP1_pp_6_2;

integer AP1_pp_7_2;

integer Exact_42_pp_8_2;

integer Exact_42_pp_9_2;

integer Exact_42_pp_10_2;

integer Exact_42_pp_11_2;

integer Exact_42_pp_12_2;

integer Full_adder_pp_13_2;

initial begin

    Half_adder_pp_2_2 = $fopen("2_Half_adder_pp_2.txt","w");

	AP1_pp_3_2 = $fopen("2_AP1_pp_3.txt","w");

    AP1_pp_4_2 = $fopen("2_AP1_pp_4.txt","w");

	AP1_pp_5_2 = $fopen("2_AP1_pp_5.txt","w");

	AP1_pp_6_2 = $fopen("2_AP1_pp_6.txt","w");

	AP1_pp_7_2 = $fopen("2_AP1_pp_7.txt","w");

	Exact_42_pp_8_2 = $fopen("2_Exact_42_pp_8.txt","w");

	Exact_42_pp_9_2 = $fopen("2_Exact_42_pp_9.txt","w");

	Exact_42_pp_10_2 = $fopen("2_Exact_42_pp_10.txt","w");

	Exact_42_pp_11_2 = $fopen("2_Exact_42_pp_11.txt","w");

	Exact_42_pp_12_2 = $fopen("2_Exact_42_pp_12.txt","w");

	Full_adder_pp_13_2 = $fopen("2_Full_adder_pp_13.txt","w");

end

always @(posedge clk) begin

    $fdisplay(Half_adder_pp_2_2, "%b",{pp_2[0],pp_2[1]});

	$fdisplay(AP1_pp_3_2, "%b",{pp_3[0],pp_3[1],pp_3[2],pp_3[3]});

	$fdisplay(AP1_pp_4_2, "%b",{pp_4[0],pp_4[1],pp_4[2],pp_4[3]});

	$fdisplay(AP1_pp_5_2, "%b",{pp_5[0],pp_5[1],pp_5[2],pp_5[3]});

	$fdisplay(AP1_pp_6_2, "%b",{pp_6[0],pp_6[1],pp_6[2],pp_6[3]});

    $fdisplay(AP1_pp_7_2, "%b",{pp_7[0],pp_7[1],pp_7[2],pp_7[3]});

	$fdisplay(Exact_42_pp_8_2, "%b",{pp_8[0],pp_8[1],pp_8[2],pp_8[3],gnd_const});

	$fdisplay(Exact_42_pp_9_2, "%b",{pp_9[0],pp_9[1],pp_9[2],pp_9[3],pp_8_exa_c2});

	$fdisplay(Exact_42_pp_10_2, "%b",{pp_10[0],pp_10[1],pp_10[2],pp_10[3],pp_9_exa_c2});

	$fdisplay(Exact_42_pp_11_2, "%b",{pp_11[0],pp_11[1],pp_11[2],pp_11[3],pp_10_exa_c2});

	$fdisplay(Exact_42_pp_12_2, "%b",{pp_12[0],pp_12[1],pp_12[2],pp_12[3],pp_11_exa_c2});

	$fdisplay(Full_adder_pp_13_2, "%b",{pp_13[0],pp_13[1],pp_12_exa_c2});

end
`endif
endmodule


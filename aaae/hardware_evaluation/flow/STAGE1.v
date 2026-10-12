`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2023/02/23 15:15:31
// Design Name: 
// Module Name: STAGE1
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


module STAGE1(
 input [7:0] A,
 input [7:0] B,
 input clk,
 output pp_0,
 output [1:0] pp_1,
 output [2:0] pp_2,
 output [3:0] pp_3, 
 output [3:0] pp_4,
 output [3:0] pp_5,
 output [3:0] pp_6,
 output [3:0] pp_7,
 output [3:0] pp_8,
 output [3:0] pp_9,
 output [3:0] pp_10,
 output [3:0] pp_11,
 output [3:0] pp_12,
 output [1:0] pp_13,
 output  pp_14           
    );

    wire cp_next;
    wire cp_now;
    wire pp_4_s;
    wire pp_4_c;
    wire pp_5_s;
    wire pp_5_c;
    wire pp_6_pro_s;
    wire pp_6_pro_c;
    wire pp_6_exa_s;
    wire pp_6_exa_c;
    wire pp_7_pro_s1;
    wire pp_7_pro_c1;
    wire pp_7_pro_s2;
    wire pp_7_pro_c2; 
    
    wire pp_8_exa_s1;
    wire pp_8_exa_c1;
    wire pp_8_exa_c2;       
    wire pp_8_exa_s2;
    wire pp_8_exa_c3;
    wire pp_8_exa_c4;
     
    wire pp_9_exa_s1;
    wire pp_9_exa_c1;
    wire pp_9_exa_c2;   
    wire pp_9_exa_s2;
    wire pp_9_exa_c3;
  
    wire pp_10_exa_c1;
    wire pp_10_exa_c2;
    wire pp_10_exa_s;
    
    wire pp_11_exa_s;
    wire pp_11_exa_c;
    wire gnd_const;

    LOGIC0_X1 u_logic0_stage1 (
        .Z(gnd_const)
    );
    
    assign pp_0 = A[0] & B[0];
    assign pp_1 = {A[0] & B[1],A[1] & B[0]};
    assign pp_2 = {A[0] & B[2],A[1] & B[1],A[2] & B[0]};
    assign pp_3 = {A[0] & B[3],A[1] & B[2],A[2] & B[1],A[3] & B[0]};
    
    assign pp_4 = {A[0] & B[4],A[1] & B[3],A[2] & B[2],pp_4_s};
    assign pp_5 = {A[0] & B[5],A[1] & B[4],pp_4_c,pp_5_s};    
    assign pp_6 = {A[0] & B[6],pp_6_exa_s,pp_5_c,pp_6_pro_s};
    
    assign pp_7 = {pp_6_exa_c,pp_6_pro_c,pp_7_pro_s2,pp_7_pro_s1};
    assign pp_8 = {pp_7_pro_c2,pp_8_exa_s2,pp_7_pro_c1,pp_8_exa_s1};
    assign pp_9 = {pp_8_exa_c3,pp_9_exa_s2,pp_8_exa_c1,pp_9_exa_s1};
    assign pp_10 = {A[3] & B[7],pp_9_exa_c3,pp_9_exa_c1,pp_10_exa_s};    
    assign pp_11 = {A[4] & B[7],A[5] & B[6],pp_10_exa_c1,pp_11_exa_s};
    assign pp_12 = {A[5] & B[7],A[6] & B[6],A[7] & B[5],pp_11_exa_c};
    assign pp_13 = {A[6] & B[7],A[7] & B[6]};
    assign pp_14 = {A[7] & B[7]};

    ///
    Half_adder Half_adder_pp_4
    (
        .a(A[4] & B[0]),
        .b(A[3] & B[1]),
        .S(pp_4_s),
        .C(pp_4_c)
    );
    AP1 AP1_pp_5
    (
        .a(A[5] & B[0]),
        .b(A[4] & B[1]),
        .c(A[3] & B[2]),
        .d(A[2] & B[3]),
        .S(pp_5_s),
        .C(pp_5_c)
    );
    ///
    AP1 AP1_pp_6
    (
        .a(A[6] & B[0]),
        .b(A[5] & B[1]),
        .c(A[4] & B[2]),
        .d(A[3] & B[3]),
        .S(pp_6_pro_s),
        .C(pp_6_pro_c)
    );
    Half_adder Half_adder_pp_6
    (
        .a(A[2] & B[4]),
        .b(A[1] & B[5]),
        .S(pp_6_exa_s),
        .C(pp_6_exa_c)
    );
    ///
    AP1 AP1_pp_7_1
    (
        .a(A[7] & B[0]),
        .b(A[6] & B[1]),
        .c(A[5] & B[2]),
        .d(A[4] & B[3]),
        .S(pp_7_pro_s1),
        .C(pp_7_pro_c1)
    );



    AP1 AP1_pp_7_2
    (
        .a(A[3] & B[4]),
        .b(A[2] & B[5]),
        .c(A[1] & B[6]),
        .d(A[0] & B[7]),
        .S(pp_7_pro_s2),
        .C(pp_7_pro_c2)
    );
    ///
    E_4_2 Exact_42_pp_8_1
    (
        .a(A[7] & B[1]),
        .b(A[6] & B[2]),
        .c(A[5] & B[3]),
        .d(A[4] & B[4]),
        .CIN(gnd_const),
        .S(pp_8_exa_s1),
        .C(pp_8_exa_c1),
        .COUT(pp_8_exa_c2)
    );
    
     Full_adder Full_adder_pp_8_2
    (
        .a(A[3] & B[5]),
        .b(A[2] & B[6]),
        .CIN(A[1] & B[7]),
        .S(pp_8_exa_s2),
        .C(pp_8_exa_c3)
    );

    ///
    E_4_2 Exact_42_pp_9
    (
        .a(A[7] & B[2]),
        .b(A[6] & B[3]),
        .c(A[5] & B[4]),
        .d(A[4] & B[5]),
        .CIN(pp_8_exa_c2),
        .S(pp_9_exa_s1),
        .C(pp_9_exa_c1),
        .COUT(pp_9_exa_c2)
    );
    Half_adder Half_adder_pp_9
    (
        .a(A[3] & B[6]),
        .b(A[2] & B[7]),
        .S(pp_9_exa_s2),
        .C(pp_9_exa_c3)
    );
    ///
    E_4_2 Exact_42_pp_10
    (
        .a(A[7] & B[3]),
        .b(A[6] & B[4]),
        .c(A[5] & B[5]),
        .d(A[4] & B[6]),
        .CIN(pp_9_exa_c2),
        .COUT(pp_10_exa_c2),
        .S(pp_10_exa_s),
        .C(pp_10_exa_c1)
    );
    ///
    Full_adder Full_adder_pp_11
    (
        .a(A[7] & B[4]),
        .b(A[6] & B[5]),
        .CIN(pp_10_exa_c2),
        .S(pp_11_exa_s),
        .C(pp_11_exa_c)
    );

`ifdef SIM
integer Half_adder_pp_4_1;

integer AP1_pp_5_1;

integer AP1_pp_6_1;

integer Half_adder_pp_6_1;

integer AP1_pp_7_1_1;

integer AP1_pp_7_2_1;

integer Exact_42_pp_8_1_1;

integer Full_adder_pp_8_2_1;

integer Exact_42_pp_9_1;

integer Half_adder_pp_9_1;

integer Exact_42_pp_10_1;

integer Full_adder_pp_11_1;

initial begin

    Half_adder_pp_4_1 = $fopen("1_Half_adder_pp_4.txt","w");

	AP1_pp_5_1 = $fopen("1_AP1_pp_5.txt","w");

	AP1_pp_6_1 = $fopen("1_AP1_pp_6.txt","w");

	Half_adder_pp_6_1 = $fopen("1_Half_adder_pp_6.txt","w");

	AP1_pp_7_1_1 = $fopen("1_AP1_pp_7_1.txt","w");

	AP1_pp_7_2_1 = $fopen("1_AP1_pp_7_2.txt");

	Exact_42_pp_8_1_1 = $fopen("1_Exact_42_pp_8_1.txt","w");

	Full_adder_pp_8_2_1 = $fopen("1_Full_adder_pp_8_2.txt","w");

	Exact_42_pp_9_1 = $fopen("1_Exact_42_pp_9.txt","w");

	Half_adder_pp_9_1 = $fopen("1_Half_adder_pp_9.txt","w");

	Exact_42_pp_10_1 = $fopen("1_Exact_42_pp_10.txt","w");

	Full_adder_pp_11_1 = $fopen("1_Full_adder_pp_11.txt","w");

end

always @(posedge clk) begin

    $fdisplay(Half_adder_pp_4_1, "%b",{A[4] & B[0],A[3] & B[1]});

	$fdisplay(AP1_pp_5_1, "%b",{A[5] & B[0],A[4] & B[1],A[3] & B[2],A[2] & B[3]});

	$fdisplay(AP1_pp_6_1, "%b",{A[6] & B[0],A[5] & B[1],A[4] & B[2],A[3] & B[3]});

	$fdisplay(Half_adder_pp_6_1, "%b",{A[2] & B[4],A[1] & B[5]});

	$fdisplay(AP1_pp_7_1_1, "%b",{A[7] & B[0],A[6] & B[1],A[5] & B[2],A[4] & B[3]});

	$fdisplay(AP1_pp_7_2_1, "%b",{A[3] & B[4],A[2] & B[5],A[1] & B[6],A[0] & B[7]});

	$fdisplay(Exact_42_pp_8_1_1, "%b",{A[7] & B[1],A[6] & B[2],A[5] & B[3],A[4] & B[4],gnd_const});

	$fdisplay(Full_adder_pp_8_2_1, "%b",{A[3] & B[5],A[2] & B[6],A[1] & B[7]});

	$fdisplay(Exact_42_pp_9_1, "%b",{A[7] & B[2],A[6] & B[3],A[5] & B[4],A[4] & B[5],pp_8_exa_c2});

	$fdisplay(Half_adder_pp_9_1, "%b",{A[3] & B[6],A[2] & B[7]});

	$fdisplay(Exact_42_pp_10_1, "%b",{A[7] & B[3],A[6] & B[4],A[5] & B[5],A[4] & B[6],pp_9_exa_c2});

	$fdisplay(Full_adder_pp_11_1, "%b",{A[7] & B[4],A[6] & B[5],pp_10_exa_c2});

end
`endif
endmodule

`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2023/02/23 15:10:34
// Design Name: 
// Module Name: Full_adder
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


module Full_adder(
  input a,
  input b,
  input CIN,
  output S,
  output C
    );
    wire S1 , T1, T2, T3;
    
    xor x1 (S1,a,b);
    xor x2 (S,S1,CIN);
    
    and A1 (T3,a,b);
    and A2 (T2,b,CIN);
    and A3 (T1,a,CIN);
    
    or O1 (C,T1,T2,T3);
endmodule

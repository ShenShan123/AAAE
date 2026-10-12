`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2023/02/23 15:09:34
// Design Name: 
// Module Name: E_4_2
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


module E_4_2(
  input a,
  input b,
  input c,
  input d,
  input CIN,
  output S,
  output C,
  output COUT
    );
    
        wire part;
    assign part = a ^ b ^ c ^ d;
    assign S = part ^ CIN;
    assign COUT = ((a ^ b) & c) |  (~(a ^ b) & a);
    assign C = (part & CIN) | (~part & d);
endmodule

`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2022/04/04 16:08:21
// Design Name: 
// Module Name: Pro
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


module sabetz(
    input a,
    input b,
    input c,
    input d,
    output S,
    output C
    );
    
    wire vdd_const;
    LOGIC1_X1 u_logic1_sabetz (
        .Z(vdd_const)
    );

    assign S = vdd_const;
    assign C = (d & (a | c)) | (a & c);
endmodule

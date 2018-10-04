%
% store2DToKaras.m
%
% Stores given matrix to a .karas file.
%
% Created by Dimitrios Karageorgiou,
%   for course "Parallel And Distributed Systems".
%   Electrical and Computers Engineering Department, AuTh, GR - 2017-2018
%
function store2DToKaras(fn, A)
    f = fopen(fn, 'w');
    fwrite(f, size(A, 1), 'uint32');
    fwrite(f, size(A, 2), 'uint32');
    fwrite(f, A', 'double');  % store in row-major order  
    fclose(f);
end
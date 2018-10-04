%
% imgKarasShow.m
%
% Loads and shows a grayscale image stored in a .karas container.
%
% Created by Dimitrios Karageorgiou,
%   for course "Parallel And Distributed Systems".
%   Electrical and Computers Engineering Department, AuTh, GR - 2017-2018
%
function imgKarasShow(fn)
    imData = load2DFromKaras(fn);        
    figure('Name', 'Filtered image');
  	imagesc(imData); axis image;
    colormap gray;
end
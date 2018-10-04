%
% generateAnlmFilteredKaras.m
%
% Denoises a given image unsing Adaptive Non Local Means algorithm.
%
% Parameters:
%   -noisy_fn : Path to a .karas file containing the image.
%   -filtered_fn : Path to a .karas file where filtered image will be
%           written.
%
% Created by Dimitrios Karageorgiou,
%   for course "Parallel And Distributed Systems".
%   Electrical and Computers Engineering Department, AuTh, GR - 2017-2018
%
% Based on code from demo_non_local_means.m written by:
%   Dimitris Floros (fcdimitr@auth.gr)
%
function generateAnlmFilteredKaras(noisy_fn, filtered_fn)
  %% PARAMETERS
    
  % number of regions
  nLevel = 6;
  
  % filter sigma value
  patchSize = [5 5];
  patchSigma = 5/3;
  
  %% LOAD NOISY IMAGE
  J = load2DFromKaras(noisy_fn);
  
  %% PREPARE ADAPTIVE NLM INPUTS
  
  % distinct levels
  L = round( J .* (nLevel-1) );
  
    store2DToKaras('woman_blonde_tiny_ids.karas', L);
  
  % adaptive sigma
  adFiltSigma = zeros( size( J ) );
  
  % sigma in each region (STD of each region)
  for i = 0 : (nLevel-1)
    adFiltSigma( L == i ) = std( J( L == i ) );
  end
  
    store2DToKaras('woman_blonde_tiny_std.karas', adFiltSigma);
  
%   % visualize inputs adaptive NLM inputs
%   figure('Name', 'Irregular search regions');
%   imagesc(L); axis image;
%   colormap parula;
  
%   figure('Name', 'Adaptive sigma value');
%   imagesc(adFiltSigma); axis image;
%   colormap parula;
    
  %% ADAPTIVE NON LOCAL MEANS
    
  tic;
  Ia = adaptNonLocalMeans( J, L, patchSize, ...
                           adFiltSigma, patchSigma );
  toc
  
  %% EXPORT FILTERED IMAGE
  store2DToKaras(filtered_fn, Ia);
end
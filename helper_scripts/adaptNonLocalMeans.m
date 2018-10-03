function If = adaptNonLocalMeans(I, L, patchSize, filtSigma, patchSigma)
% ADAPTNONLOCALMEANS - Adaptive Non local means
% 
%  Adaptive nonlocal means[1] with irregular search windows.
%   
% SYNTAX
%
%   IF = ADAPTNONLOCALMEANS( IN, LL, FILTSIGMA, PATCHSIGMA )
%
% INPUT
%
%   IN          Input image                     [m-by-n]
%   LL          Search regions                  [m-by-n]
%   PATCHSIZE   Neighborhood size in pixels     [1-by-2]
%   FILTSIGMA   Filter sigma value              [m-by-n]
%   PATCHSIGMA  Patch sigma value               [scalar]
%
% OUTPUT
%
%   IF          Filtered image after nlm        [m-by-n]
%
% DESCRIPTION
%
%   IF = ADAPTNONLOCALMEANS(IN,LL,PATCHSIZE,FILTSIGMA,PATCHSIGMA) 
%   applies non local means algorithm with sigma value of
%   FILTSIGMA(i,j) at each pixel, using a Gaussian patch of size
%   PATCHSIZE with sigma value of PATCHSIGMA.  Each region LL is
%   denoised independently.
%
% REFERENCES
% 
%   [1] Antoni Buades, Bartomeu Coll, and J-M Morel. A non-local
%   algorithm for image denoising. In CVPR 2005, volume 2, pages
%   60–65, 2005
%
  
  
  %% USEFUL FUNCTIONS
  
  % create 3-D cube with local patches
  patchCube = @(X,w) ...
      permute( ...
          reshape( ...
              im2col( ...
                  padarray( ...
                      X, ...
                      (w-1)./2, 'symmetric'), ...
                  w, 'sliding' ), ...
              [prod(w) size(X)] ), ...
          [2 3 1] );
  
  
  %% PRE-PROCESS
  
  % create 3D cube
  B = patchCube(I, patchSize);
  [m, n, d] = size( B );
  B = reshape(B, [ m*n d ] );
  
  % Gaussian patch
  H = fspecial('gaussian',patchSize, patchSigma);
  H = H(:) ./ max(H(:));
  
  % apply gaussian patch on 3D cube
  B = bsxfun( @times, B, H' );
  
  
  %% REGION GENERATION
  
  % list of region IDs
  regionIds = unique( L );

  % number of regions
  nRegion = length( regionIds );
  
  % prepare cell array with pixels in each region
  regionPix = cell( nRegion, 1 );
  
  for iRegion = 1 : nRegion
    
    regionPix{iRegion} = find( L == regionIds( iRegion ) );
    
  end
  
  %% FILTER EACH REGION SEPARATELY

  % final image
  If = zeros( size(I) );
  
  for iRegion = 1 : nRegion
    
    % working set of pixels
    workPixels = regionPix{iRegion};
    
    % compute kernel (in current region)
    D = squareform( pdist( B(workPixels,:), 'euclidean' ) );
    D = exp( -D.^2 ./ filtSigma(workPixels).^2 );
    D(1:length(D)+1:end) = max(max(D-diag(diag(D)),[],2), eps);
    
    % generate filtered image
    If(workPixels) = D*I(workPixels) ./ sum(D, 2);
    
  end
    
end


%%------------------------------------------------------------
%
% AUTHORS
%
%   Dimitris Floros                         fcdimitr@auth.gr
%
% VERSION
%
%   0.1 - January 10, 2018
%
% CHANGELOG
%
%   0.1 (Dec 28, 2016) - Dimitris
%       * initial implementation (based on v0.2 of nonLocalMeans.m)
%
% ------------------------------------------------------------


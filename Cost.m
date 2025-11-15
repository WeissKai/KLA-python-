

%__________________________________________________________________     %
%                  Kirchhoff’s law algorithm (KLA)                      %
%                                                                       %
%                                                                       %
%                  Developed in MATLAB R2024b (MacOs)                   %
%                                                                       %
%                      Author and programmer                            %
%                                                                       %
%                ---------------------------------                      %
%                          Mojtaba Ghasemi                              %
%     Co:   Nima Khodadadi (ʘ‿ʘ) University of California Berkeley      %
%                             e-Mail                                    %
%                ---------------------------------                      %
%                      Nimakhan@berkeley.edu                            %
%                                                                       %
%                                                                       %
%                            Homepage                                   %
%                ---------------------------------                      %
%                    https://nimakhodadadi.com                          %
%                                                                       %
%                                                                       %
%                                                                       %
%                                                                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                           Citation                                    %
% Ghasemi, M, Khodadadi, N. et al.                                      %
% Kirchhoff’s law algorithm (KLA): a novel physics-inspired             %
% non-parametric metaheuristic algorithm for optimization problems      %
% Artificial Intelligence Review.                                       %
% https://doi.org/10.1007/s10462-025-11289-5                            %
% ----------------------------------------------------------------------%
function z=Cost(x, jj)
 
 
 [ps,D]=size(x);


% 1. Basic Shifted Sphere Function
if jj==1

x=x-repmat(ps,1);
z=sum(x.^2,2);
end

% 	2. Basic Schwefel's Problem 1.2 
 if jj==2

x=x-repmat(ps,1);
  z=0;
for i=1:D
z=z+sum((x(:,1:i)),2).^2;

end

end



% %	3. Basic Schwefel's Problem 1.2 with Noise in Fitness 
if jj==3 

 x=x-repmat(ps,1);
z=0;
for i=1:D
    z=z+sum(x(:,1:i),2).^2;
end
z=z.*(1+0.4.*abs(normrnd(0,1,ps,1)));
end

end




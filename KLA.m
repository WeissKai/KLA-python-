

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
clc;
clear all
for Nf=1:3    %%% Number of the test function

for iijj=1:2  %%% Number of independent executions of each test function

%%% Problem Definition

 CostFunction=@(x, Nf) Cost(x, Nf);        %%% The test function

nVar=30;                 %%% Number of Decision Variables
func_num=Nf;             %%% if you want to use form  Cost Functions CEC 2014 or other CEC
VarSize=[1 nVar];        %%% Decision Variables Matrix Size

VarMin=-100;             %%% Decision Variables Lower Bound
VarMax= -VarMin;             %%% Decision Variables Upper Bound
ebs=realmin;
%%% KLA Parameters; KLA: A non-parametric physics-based inspired algorithm for optimization

 MaxIt=3000;        %%% Maximum Number of Function Evoulations
nPop=50;            %%% Number of Solutions (Swarm Size)
%%% Iter=5000;         %%% Maximum Number of Iterations  
%%% MaxIt=Iter*nPop;        %%% Maximum Number of Function Evoulations



%%% Empty Population Structure
solutions.Position=[];
solutions.Cost=[];

%%% Initialize Population Array
pop=repmat(solutions,nPop,1);

%%% Initialize Best Solution Ever Found
BestSol.Cost=inf;
TT=-inf;
%%% Create Initial Population
for i=1:nPop
   pop(i).Position=unifrnd(VarMin,VarMax,VarSize);
   pop(i).Cost=CostFunction(pop(i).Position,func_num);
%%% pop(i).Cost=feval(fhd,pop(i).Position',func_num);%if you want to use form  Cost Functions CEC 2014
   if pop(i).Cost<=BestSol.Cost
       BestSol=pop(i);
   end
    if pop(i).Cost>TT
       TT=pop(i).Cost;
    end
    BestCost1(i)=BestSol.Cost;
end

%%% Array to Hold Best Cost Values
BestCost=zeros(MaxIt,1);

%%% KLA Main Loop
 it=nPop;
%%% for it=1:Iter %%% If you want to use it in terms of repetition.
while it<=MaxIt

 newpop=repmat(solutions,nPop,1);
    for i=1:nPop
        newpop(i).Cost = inf;


 A=randperm(nPop);
        
        A(A==i)=[];
        
        a=A(1);
        b=A(2);
        jj=A(3);
            
            
q=((pop(i).Cost-pop(jj).Cost)+ebs)/(abs((pop(i).Cost-pop(jj).Cost))+ebs);
Q=(pop(i).Cost-pop(a).Cost)/(abs((pop(i).Cost-pop(a).Cost))+ebs);
Q2=(pop(i).Cost-pop(b).Cost)/(abs((pop(i).Cost-pop(b).Cost))+ebs);


q1=((pop(jj).Cost)/(pop(i).Cost))^(2*rand);
Q1=((pop(a).Cost)/(pop(i).Cost))^(2*rand);
Q21=((pop(b).Cost)/(pop(i).Cost))^(2*rand);


S1=q1*q*rand(VarSize).*(pop(jj).Position-pop(i).Position);
S2=Q*Q1*rand(VarSize).*(pop(a).Position-pop(i).Position);
S3=Q2*Q21*rand(VarSize).*(pop(b).Position-pop(i).Position);
S=(rand+rand)*S1+(rand+rand)*S2+(rand+rand)*S3;


newsol.Position =pop(i).Position+S;

                newsol.Position=max(newsol.Position,VarMin);
                newsol.Position=min(newsol.Position,VarMax);
                
                newsol.Cost=CostFunction(newsol.Position,func_num);
%%% newsol.Cost=feval(fhd,newsol.Position',func_num); %%%if you want to use form  Cost Functions CEC 2014 or other CEC

              if newsol.Cost<=pop(i).Cost
                pop(i) = newsol;
                
                 if pop(i).Cost<=BestSol.Cost
                 BestSol=pop(i);
                 end
            
              end
   it=it+1;
          BestCost1(it)=BestSol.Cost;  
        end

   
   
   %%% Store Best Cost Ever Found
    BestCost(it)=BestSol.Cost;
    
    %%% Show Iteration Information
    disp(['Iteration ' num2str(it) ': Best Cost = ' num2str(BestCost(it))]);
    

end

Cost_Rsult(1, iijj)=BestCost(end);
Rsult(iijj, :)=BestCost1;
end

Nf
Mmean(Nf)=mean(Cost_Rsult)
Bbest(Nf)=min(Cost_Rsult)
Std(Nf)=std(Cost_Rsult)
 hold on
plot(log(mean(Rsult)),'k','LineWidth',4); hold on
end


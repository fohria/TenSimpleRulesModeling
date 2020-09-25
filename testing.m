%% iterate simulation and fitting
clear;
% real simulation parameters
realalpha=.1;
realbeta=8;
realrho=.9;
realK=4;

options=optimset('MaxFunEval',100000,'Display','off','algorithm','active-set');%


iter=1;
ninitialpoints=10;
[stim,update,choice,rew,setsize]=simulate(realalpha,realbeta,realrho,realK);
pars=[];
% fit simulated data with ninitialpoints random starting points
for init=1:ninitialpoints
    disp(['ITERATION NUMBER', num2str(init)])
    x0=rand(1,3);
    [pval,fval,bla,bla2] =fmincon(@(x) computellh(x,realK,stim,update,choice,rew,setsize),x0,[],[],[],[],...
        [0 0 0],[1 1 1],[],options);
    pars(init,:) = [pval,fval]
    [m,i]=min(pars(:,end))
    bestllh(iter,init)=m
    bestpars(iter,init,:)=pars(i,1:end-1)
%     disp(['pars is now: ', mat2str(pars)])
%     disp(['bestllh is now: ', mat2str(bestllh)])
%     disp(['bestpars is now: ', mat2str(bestpars)])
end

%%
clear;
% real simulation parameters
realalpha=.1;
realbeta=8;
realrho=.9;
realK=4;

options=optimset('MaxFunEval',100000,'Display','off','algorithm','active-set');
% number of random starting points for optimizer
ninitialpoints=10;
% for 100 simulations
for iter = 1:10
    disp(['simulation #',num2str(iter)])
    % generate data
    [stim,update,choice,rew,setsize]=simulate(realalpha,realbeta,realrho,realK);
    pars=[];
    % fit simulated data with ninitialpoints random starting points
    for init=1:ninitialpoints
        x0=rand(1,3);
        [pval,fval,bla,bla2] =fmincon(@(x) computellh(x,realK,stim,update,choice,rew,setsize),x0,[],[],[],[],...
            [0 0 0],[1 1 1],[],options);
        pars(init,:) = [pval,fval];
        [m,i]=min(pars(:,end));
        bestllh(iter,init)=m;
        bestpars(iter,init,:)=pars(i,1:end-1);
    end
    % find global best fit
    % hmm i'm not sure this works as intended, if all or some of the fits,
    % i.e. if values in pars(:,end) are the same, the index received
    % (i variable in below line) becomes 4?? next time 9. so it seems it
    % picks a random index if it finds several that are the same, so all
    % the graphs below cant be trusted. unless i'm missing something which
    % of course is likely
    [mf,i]=min(pars(:,end));
    % find at which random starting point it was found
    when(iter,1)=i;
    % find at which random starting point a likelihood within .01 of the
    % global best was found
    i=find(bestllh(iter,:)<bestllh(iter,end)+.01);
    when(iter,2)=i(1);
    % find at which random starting point a likelihood within .1 of the
    % global best was found
    i=find(bestllh(iter,:)<bestllh(iter,end)+.1);
    when(iter,3)=i(1);
end
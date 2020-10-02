addpath('./SimulationFunctions')
addpath('./AnalysisFunctions')
addpath('./HelperFunctions')
addpath('./FittingFunctions')
addpath('./LikelihoodFunctions')


%%
clear

actions = [1 1 0 1 1 1 1 1 1 1 1 0 0 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 0 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 0 1 1 0 1 1 1 0 0 1 0 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 0 0 0 1 0 1 1 0 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 0 1 1 1 1 1 0 1 1 0 1 1 1 1 1 1 1 1 1 0 1 1 0 0 0 1 1 1 1 0 0 1 0 0 1 1 0 1 1 0 1 0 1 1 1 1 1 1 1 1 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 0 1 1 0 0 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 0 1 1 1 0 1 0 0 1 0 1 0 1 1 1 0 1 1 1 1 0 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 0 1 0 1 1 1 1 0 0 1 1 1 0 1 1 1 1 1 1 1 1 0 1 0 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 0 1 1 0 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 0 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 0 0 1 0 0 1 1 1 0 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 1 1 1 1 1 1 1 1 1 1 0 1 0 1 1 1 1 1 0 1 0 1 0 1 1 1 0 1 0 0 1 1 0 1 0 1 1 1 0 1 0 1 1 0 1 1 1 0 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 0 1 1 1 1 1 0 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 0 1 1 1 1 1 1 0 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 0 1 1 1 0 0 1 1 1 0 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 0 0 1 1 0 1 0 1 0 1 1 1 1 0 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 0 1 1 1 1 1 0 1 1 1 0 1 1 1 1 0 1 0 1 1 0 0 1 1 1 1 1 0 1 1 0 1 1 1 0 1 1 0 1 1 1 0 0 1 1 1 1 1 1 0 1 0 0 1 1 0 0 0 0 1 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 0 1 0 1 1 1 1 1 1 1 1 1 1 1 1 0];
for i=1:length(actions)
    if actions(i) == 0
        actions(i) = 1;
    else
        actions(i) = 2;
    end
end
rewards = [1 1 0 1 1 1 1 1 1 1 1 1 0 0 0 1 1 1 0 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 0 1 0 0 1 1 1 1 1 1 1 1 1 1 1 0 1 1 0 1 0 1 1 1 1 1 1 0 1 1 0 1 1 1 1 1 1 1 1 1 0 0 1 1 1 0 1 0 1 0 1 1 1 0 0 0 0 1 1 1 1 0 1 1 0 1 1 0 1 1 1 1 1 0 0 0 1 1 1 0 1 0 0 1 0 1 1 0 1 1 1 1 1 0 1 1 1 0 0 1 1 1 0 1 1 1 1 1 1 1 1 0 1 1 1 1 1 0 1 1 1 1 0 1 0 1 1 1 0 1 1 1 0 1 1 1 0 1 1 1 1 0 0 0 1 0 1 1 1 0 1 1 1 1 1 0 1 1 1 0 0 0 0 1 1 1 1 1 0 0 1 1 1 1 0 0 0 0 1 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 1 0 1 1 1 0 0 1 0 1 1 1 1 1 0 0 1 0 0 0 1 1 1 1 1 0 0 1 0 1 1 0 1 1 1 1 0 0 0 1 0 0 0 0 1 0 1 1 0 0 1 0 0 0 1 0 1 0 1 1 1 1 0 1 1 1 1 1 1 0 1 1 1 1 1 1 1 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 0 1 1 1 0 1 0 1 1 1 1 1 1 1 1 0 0 1 1 1 0 1 1 0 1 1 0 0 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 0 1 1 1 1 0 0 1 1 1 1 1 1 0 1 1 1 1 1 0 1 0 1 1 1 1 1 1 0 1 1 1 1 1 1 0 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 1 1 1 1 0 1 0 1 0 1 1 0 0 1 1 1 0 1 0 1 1 1 1 0 0 0 1 0 1 1 1 1 1 1 1 1 1 0 1 1 1 1 0 0 1 1 0 1 0 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 0 1 0 0 0 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 1 1 1 1 0 1 1 1 0 0 1 1 1 1 0 1 1 1 0 1 0 0 1 0 1 0 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 0 0 0 0 0 1 1 0 1 1 1 0 1 1 1 1 1 0 1 1 1 0 1 0 1 1 1 1 1 1 1 1 1 0 0 1 0 0 0 1 1 1 1 0 1 0 1 1 1 0 1 0 0 1 1 0 1 1 0 0 1 1 1 1 0 0 1 0 0 1 1 1 1 0 1 1 0 0 1 1 1 1 0 0 0 1 0 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 0 0 0 0 0 1 1 1 1 0 1 0 1 1 1 0 1 0 0 1 0 1 1 1 1 1 1 1 0 1 1 1 0 0 1 0 0 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 0 1 1 1 0 1 1 1 0 0 0 1 1 1 1 0 0 0 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 0 1 0 0 0 1 1 1 1 1 0 0 0 1 1 0 1 1 1 1 1 1 1 1 1 1 0 0 1 1 1 1 1 1 1 1 1 1 1 0 1 0 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 0 1 0 0 0 1 1 1 0 1 0 1 1 0 1 0 0 0 1 0 1 1 1 0 1 0 1 1 1 1 1 1 1 0 1 1 1 0 1 1 1 0 0 0 0 1 1 1 1 1 1 1 0 1 1 1 0 1 0 1 1 0 1 0 1 0 1 1 0 1 0 0 0 0 0 0 0 1 1 0 0 0 0 0 1 1 1 1 0 0 1 0 0 1 1 1 1 1 1 1 1 1 0 1 1 0 1 1 1 1 1 0 1 1 1 1 0 1 1 1 0 1 0 1 0 1 1 1 1 1 0 1 1 0 1 1 1 1];
rewards = logical(rewards);

alpha = 0.2;
beta = 5;
alpha_c = 0.4;
beta_c = 3;

bias = 0.2;
epsilon=0.2;

lik_M1random_v1(actions, rewards, bias)
lik_M2WSLS_v1(actions, rewards, epsilon)
lik_M3RescorlaWagner_v1(actions, rewards, alpha, beta)
lik_M4CK_v1(actions, rewards, alpha_c, beta_c)
lik_M5RWCK_v1(actions, rewards, alpha, beta, alpha_c, beta_c)

% lik_M3RescorlaWagner_v1(actions, rewards, alpha, beta)
% [params, loglike, BIC] = fit_M3RescorlaWagner_v1(actions, rewards)
% fit_all_v1(actions, rewards)
%%
clear
for i=1:10000
    y(i) = exprnd(1);
end
median(y)
hist(y, 20)
%%
clear
Q = [0.4 0.5];
beta = 3;

exp(beta * Q) / sum(exp(beta * Q))

%%
% test how many correct choices for MODEL3
T = 1000;
mu = [0.2 0.8];

corrects = zeros(1, T);

for i=1:1000
    alpha = 0.2;
    beta = 1.7;
    [a, r] = simulate_M3RescorlaWagner_v1(T, mu, alpha, beta);

    correct_ratio = length(a(a == 2)) / T;
    corrects(i) = correct_ratio;
end

mean(corrects)
%% 
% test how many correct choices for MODEL1
T = 1000;
mu = [0.2 0.8];

corrects = zeros(1, T);

for i=1:1000
    bias = 0.2;
    [a, r] = simulate_M1random_v1(T, mu, bias);

    correct_ratio = length(a(a == 2)) / T;
    corrects(i) = correct_ratio;
end

mean(corrects)

%%
scatter(1:length(a), a)
ylim([-0.5, 2.5])
%%
plot(1:length(a), a)
hist(a)

%%
[~, ~, BIC(1)] = fit_M1random_v1(a, r);
[~, ~, BIC(2)] = fit_M2WSLS_v1(a, r);
[~, ~, BIC(3)] = fit_M3RescorlaWagner_v1(a, r);
[~, ~, BIC(4)] = fit_M4CK_v1(a, r);
[~, ~, BIC(5)] = fit_M5RWCK_v1(a, r);

[M, iBEST] = min(BIC);
BEST = BIC == M;
BEST = BEST / sum(BEST);

%%

CM = zeros(5);

T = 1000;
mu = [0.2 0.8];

% Model 1
for count=1:5
    
    disp('ROuND::::::::::::::::::::::::::::')
    
    count
    
    disp('MODEL1')
    
    b = rand;
    [a, r] = simulate_M1random_v1(T, mu, b);
    [BIC, iBEST, BEST] = fit_all_v1(a, r);
    CM
    CM(1,:) = CM(1,:) + BEST;
    CM
    
    disp('MODEL2')
    
    epsilon = rand;
    [a, r] = simulate_M2WSLS_v1(T, mu, epsilon);
    [BIC, iBEST, BEST] = fit_all_v1(a, r);
    CM
    CM(2,:) = CM(2,:) + BEST;
    CM
end
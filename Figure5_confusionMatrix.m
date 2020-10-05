%%%%% Bob Wilson & Anne Collins
%%%%% 2018
%%%%% Code to produce figure 5 in submitted paper "Ten simple rules for the
%%%%% computational modeling of behavioral data"

% FYI this file requires installation of the optimization toolbox, in case
% you get the error "unrecognized function or variable getIpOptions":
% https://uk.mathworks.com/matlabcentral/answers/491764-unrecognized-function-or-variable-getipoptions

% furthermore, this file outputs what i believe is Box5/Figure1 B, i.e.
% the confusion matrix with beta/softmax parameters increased by 1.
% in order to get Box5/Figure1 A, we have to remove +1 on beta parameters
% in models 3-5 below

% additionally; there's a fairly large variety in the output confusion
% matrices. authors mention the variety between Figures A and B, but not
% the variety within a figure. so one should perhaps run with more than 100
% counts as they have here, or at least mention how many runs they did or
% that they ran this file many times to get the figure in the paper. or use
% a seed i guess would be another option.

clear

addpath('./SimulationFunctions')
addpath('./AnalysisFunctions')
addpath('./HelperFunctions')
addpath('./FittingFunctions')
addpath('./LikelihoodFunctions')

%%
tic
CM = zeros(5);

T = 1000;
mu = [0.2 0.8];


for count = 1:100
    count

    figure(1); clf;
    FM = round(100*CM/sum(CM(1,:)))/100;
    t = imageTextMatrix(FM);
    set(t(FM'<0.3), 'color', 'w')
    hold on;
    [l1, l2] = addFacetLines(CM);
    set(t, 'fontsize', 22)
    title(['count = ' num2str(count)]);
    set(gca, 'xtick', [1:5], 'ytick', [1:5], 'fontsize', 28, ...
        'xaxislocation', 'top', 'tickdir', 'out')
    xlabel('fit model')
    ylabel('simulated model')


    drawnow
    % Model 1
    b = rand;
    [a, r] = simulate_M1random_v1(T, mu, b);
    [BIC, iBEST, BEST] = fit_all_v1(a, r);
    CM(1,:) = CM(1,:) + BEST;

    % Model 2
    epsilon = rand;
    [a, r] = simulate_M2WSLS_v1(T, mu, epsilon);
    [BIC, iBEST, BEST] = fit_all_v1(a, r);
    CM(2,:) = CM(2,:) + BEST;

    % Model 3

    alpha = rand;
    beta = 1+exprnd(1);
    [a, r] = simulate_M3RescorlaWagner_v1(T, mu, alpha, beta);
    [BIC, iBEST, BEST] = fit_all_v1(a, r);
    CM(3,:) = CM(3,:) + BEST;

    % Model 4
    alpha_c = rand;
    beta_c = 1+exprnd(1);
    [a, r] = simulate_M4ChoiceKernel_v1(T, mu, alpha_c, beta_c);
    [BIC, iBEST, BEST] = fit_all_v1(a, r);
    CM(4,:) = CM(4,:) + BEST;

    % Model 5
    alpha = rand;
    beta = 1+exprnd(1);
    alpha_c = rand;
    beta_c = 1+exprnd(1);
    [a, r] = simulate_M5RWCK_v1(T, mu, alpha, beta, alpha_c, beta_c);
    [BIC, iBEST, BEST] = fit_all_v1(a, r);
    CM(5,:) = CM(5,:) + BEST;

end
toc
%%
figure(1);
title('')
set(gcf, 'Position', [811   417   500   400])
set(gca, 'fontsize', 28);
saveFigurePdf(gcf, '~/Figures/Figure5b')
%
%
% [Xf, LL, BIC] = fit_M1random_v1(a, r);
% [Xf, LL, BIC] = fit_M2WSLS_v1(a, r);
% [Xf, LL, BIC] = fit_M3RescorlaWagner_v1(a, r);
% [Xf, LL, BIC] = fit_M4CK_v1(a, r);
% [Xf, LL, BIC] = fit_M5RWCK_v1(a, r);

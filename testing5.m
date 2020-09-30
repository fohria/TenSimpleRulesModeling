
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
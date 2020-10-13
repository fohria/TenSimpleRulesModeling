% this function is a bit confusing for two reasons
% first, it returns loseStay not loseshift as you'd think from the model name
% second, function name is wsls but what is returned is actually lsws
function out = analysis_WSLS_v1(a, r)

aLast = [nan a(1:end-1)];
stay = aLast == a;
rLast = [nan r(1:end-1)];

winStay = nanmean(stay(rLast == 1));
loseStay = nanmean(stay(rLast == 0));
out = [loseStay winStay];

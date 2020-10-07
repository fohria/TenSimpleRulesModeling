clear;

alphaP=.3+.4*rand;
beta = 4+5*rand;
alphaN=0;
k=0;
for s=1:10
    Q=ones(3,3)/3;
    for t=1:45
        k=k+1;
        o=1+rem(t-1,3);
        corA=1+rem(o-1,2);
        p=exp(beta*Q(1,:))/sum(exp(beta*Q(1,:)));
        cdf=[0 cumsum(p)];
        a=find(cdf<rand);a=a(end);
        r=a==corA;
        alpha = r*alphaP + (1-r)*alphaN;
        Q(o,a)=Q(o,a)+alpha*(r-Q(o,a));
        D(k,:)=[s,a,r];
        P(s,t)=r;
    end
end
% P = (P(:,1:3:end)+P(:,2:3:end)+P(:,3:3:end))/3;
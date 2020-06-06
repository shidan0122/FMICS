function [ Final_Y ] = F_trans_Y( Ltest,F,c )
n = size(Ltest,1);

%% Initialize
Q=randn(c, c); Q=orth(Q);

Pre_Y = zeros(n, c);
for j=1:n
    Pre_Y(j,unidrnd(c))=1;
end

NITER = 20;
%% optimization
for iter = 1:NITER
    % Q
    Q=GPI((F'*F),F'*Pre_Y,1);
    % Y
    Pre_Y = zeros(n, c);
    for i=1:n
        P=F*Q;
        [~,I1]=max(P(i,:));
        Pre_Y(i,I1)=1;
        Final_Y(i)=I1;
    end
    Final_Y=Final_Y';
end



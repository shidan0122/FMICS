function [ F ] = FMSCS(X,fea,c,param)
% Flexible Multi-view Image Clustering with Self-adaptive

view_num = size(X,2);
[n, ~] = size(fea);
maxIter = 30;
%% Normalization
for i = 1 : view_num
    for  j = 1:n
        X{i}(j,:) = ( X{i}(j,:) - mean( X{i}(j,:) ) ) / std( X{i}(j,:) ) ;
    end
end
%% =========================Initialization==========================
% S
options = [];
options.Metric = 'HeatKernel';
options.NeighborMode = 'KNN';
options.WeightMode = 'Cosine';
options.k = 14;
for i = 1:view_num
    W = constructW(X{i},options);
    A{i}.data = W;
end
temp_A = zeros(n, n);
for i=1:view_num
    temp_A = temp_A + A{i}.data;
end
S = temp_A/view_num;
S = (S + S')/2;
% F
F = rand(n,c);
F = orth(F);
% Pv
for i=1:view_num
    vdim(i) = size(X{i},2);
    Pv{i} = rand(vdim(i), c);
end
%% ==========================optimization===========================
iter = 1;
while iter<=maxIter
    % update weight
    for v = 1:view_num
        hv(v) = norm(X{v}*Pv{v}-F,'fro');
    end
    for v = 1:view_num
        alphav(v) = 0.5/norm(S-A{v}.data,'fro');
        etav{v} = hv(v)/sum(hv);
    end
    % update S
    dist = L2_distance_1(F',F');
    S = zeros(n);
    for i=1:n
        a0 = zeros(1,n);
        for v = 1:view_num
            temp = A{v}.data;
            a0 = a0+alphav(1,v)*temp(i,:);
        end
        idxa0 = find(a0>0);
        ai = a0(idxa0);
        di = dist(i,idxa0);
        ad = (ai-0.5*param.beta*di)/sum(alphav);
        S(i,idxa0) = EProjSimplex_new(ad);
    end
    % update Pv, F
    formulation_part1 = zeros(n,n);
    for v = 1:view_num
        Id{v}  = eye(vdim(v));
        G{v} = inv(1/etav{v}*X{v}'*X{v}+param.lambda*Id{v});
        Pv{v} = 1/etav{v}*G{v}*X{v}'*F;
        G1{v} = inv(X{v}'*X{v}+param.lambda*Id{v});
        formulation_part1 = formulation_part1+1/etav{v}*X{v}*G1{v}*X{v}';
    end
    S = (S+S')/2;
    D = diag(sum(S));
    L = D-S;
    Lc = L-param.mu*formulation_part1;
    [M1, lam] = eig(Lc);
    [lam, ind] = sort(diag(lam),'ascend');
    F = M1(:,ind(1:c));
    iter = iter+1;
end














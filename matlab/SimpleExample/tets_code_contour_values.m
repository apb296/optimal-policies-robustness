close all
clear Z
s=2;

z_name={'$z_{low}$','$z_{high}$'};
figure()
ctr=1;
for s=1:2
for s_star=1:1
    subplot(2,2,ctr)
    z_data=op.list_V_p((s-1)*length(op.domain)/op.N+1:(s)*length(op.domain)/op.N,s_star);



domain_fit=op.domain((s-1)*length(op.domain)/op.N+1:(s)*length(op.domain)/op.N,1:2);




Z = reshape(z_data,op.lambda_tilde_grid_size,op.mu_grid_size);
%v=[min(min(Z))*.4,0,max(max(Z))*.4];

[C,h] = contour(op.mu_grid',op.lambda_tilde_grid(s,:)',Z,'linecolor','k','linewidth',2);
xlabel('$\mu$','interpreter','latex')
ylabel('$\tilde{\lambda}$','interpreter','latex')
text_handle = clabel(C,h);
title(strcat('Iso-growth: $\mu$, $z_{t+1}$=', z_name{s_star}, ' $z_t$ = ', z_name{s}),'interpreter','latex')
ctr=ctr+1;
end
end

break
figure()
ctr=1;
for s=1:2
for s_star=1:2
    subplot(2,2,ctr)
    z_data=op.list_lambda_tilde((s-1)*length(op.domain)/op.N+1:(s)*length(op.domain)/op.N,s_star)-op.domain((s-1)*length(op.domain)/op.N+1:(s)*length(op.domain)/op.N,2);

%   z_data=op.list_pi((s-1)*length(op.domain)/op.N+1:(s)*length(op.domain)/op.N)



domain_fit=op.domain((s-1)*length(op.domain)/op.N+1:(s)*length(op.domain)/op.N,1:2);



Z = reshape(z_data,op.lambda_tilde_grid_size,op.mu_grid_size);
v=[min(min(Z))*.4,0,max(max(Z))*.4];
[C,h] = contour(op.mu_grid',op.lambda_tilde_grid(s,:)',Z,v,'linecolor','k','linewidth',2);
xlabel('$\mu$','interpreter','latex')
ylabel('$\tilde{\lambda}$','interpreter','latex')
text_handle = clabel(C,h);
title(strcat('Iso-growth: $\tilde{\lambda}$, $z_{t+1}$=', z_name{s_star}, ' $z_t$ = ', z_name{s}),'interpreter','latex')
ctr=ctr+1;
end
end
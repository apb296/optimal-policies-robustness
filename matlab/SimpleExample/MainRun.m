clear all
close all

op=OPRob();
 op.alpha_a_pi=.9*op.alpha_p_pi;
 op.alpha_a_x=1.1*op.alpha_p_x;
 
% op.alpha_p_pi=.9*op.alpha_p_pi;
% op.alpha_p_x=1*op.alpha_p_x;
% 
op.theta_a=.15000000;
op.theta_p=.15000000;

op.build_grid()
op.initialize()

op.update_policy_rules()



ppi=[]
for n=1:3
    for k=1:3

op.update_policy_rules()
op.iterate_on_value_functions()


    end
tic
for i=1:length(op.domain)                                                                                                                                             
op.get_new_policies_on_grid(i)
end
toc
op.update_policy_rules()
ppi=[ppi op.list_V_a];
end

break


% 

% 
 figure()
 plot(op.lambda_tilde_grid(1,:)',op.V_a{1}(horzcat(op.lambda_tilde_grid(1,:)'*0+mean(op.mu_grid),op.lambda_tilde_grid(1,:)')))
% 
% 
% 
% 
% 
 subplot(2,2,1)
 figure()
 plot(op.mu_grid(1,:)',op.mu{1}(horzcat(op.mu_grid',op.mu_grid'*0+mean(op.lambda_tilde_grid(1,:))'))-op.mu_grid(1,:)')
 hold on
 plot(op.mu_grid(1,:)',op.mu{2}(horzcat(op.mu_grid',op.mu_grid'*0+mean(op.lambda_tilde_grid(2,:))'))-op.mu_grid(1,:)','r')

break





for i=1:length(op.domain)
state.mu=op.domain(i,1);
state.lambda_tilde=op.domain(i,2)
state.s=op.domain(i,3)
error_pc(i)=op.error_in_pc(state)
end


break

op.iterate_on_value_functions()


for n=1:25
op.update_policy_rules()
op.iterate_on_value_functions()
op.coeff_V_a
end

plot(op.lambda_tilde_grid(1,:)',op.V_a{1}(horzcat(op.lambda_tilde_grid(1,:)'*0+mean(op.mu_grid),op.lambda_tilde_grid(1,:)')))
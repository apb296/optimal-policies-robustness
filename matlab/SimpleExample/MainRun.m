clear all
close all

op=OPRob();
op.theta_a=2000000;
op.theta_p=2000000;

op.build_grid()
op.initialize()
op.update_policy_rules()


for k=1:10
    tic
op.iterate_on_value_functions(5)
op.update_belief_distortions()
op.iterate_on_pi(5)
op.solve_state_lom_on_grid_parallel()
op.update_policy_rules()
 [op.error_coeff_pi op.error_coeff_x op.error_coeff_V_a op.error_coeff_V_p]
 toc
end
op.update_policy_rules()
op.update_belief_distortions()
max(abs(op.error_in_pc()))
max(abs(op.error_in_foc(2)))
    
    
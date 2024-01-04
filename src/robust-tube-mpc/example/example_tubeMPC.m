addpath('../src/')
addpath('../src/utils/')

% fix random seed
rng(0);

% make your own discrete linear system with disturbance
A = [1 1; 0 1];
B = [0.5; 1]; 
Q = diag([1, 1]);
R = 0.1;

W_vertex = [0.15, 0.15; 0.15, -0.15; -0.15, -0.15; -0.15, 0.15]; % construct a convex set of disturbance (2dim here)
W = Polyhedron(W_vertex);

% construct disturbance Linear system
disturbance_system = DisturbanceLinearSystem(A, B, Q, R, W);

% constraints on state Xc and input Uc
Xc_vertex = [2, -2; 2 2; -15 2; -15 -2];
Uc_vertex = [1; -1];
Xc = Polyhedron(Xc_vertex);
Uc = Polyhedron(Uc_vertex);

% create a tube_mpc simulater
% if N_horizon is too small, the path will never reach inside the robust MPI-set X_mpi_robust in time step N_horizon, then the problem becomes infeasible. 
N_horizon = 30;
w_min = [0; -0.10];
w_max = [0; 0.10];
mpc = TubeModelPredictiveControl(disturbance_system, Xc, Uc, N_horizon);

% The robust MPC guidances the path inside the robust MPI-set so that the path will reach the robust MPI-set in N_horizon. 
x = [-13; -2];
savedir_name = './results/';
mkdir(savedir_name);

for i = 1:15
    disp(i)
    u_next = mpc.solve(x);
    x = disturbance_system.propagate(x, u_next); % additive disturbance is considered inside the method 
    mpc.show_prediction();
    filename = strcat(savedir_name, 'tmpc_seq', number2string(i), '.png')
    saveas(gcf, char(filename)); % removing this line makes the code much faster
    clf;
end

